import torch
from _typeshed import Incomplete
from enum import Enum
from torch.fx import Node

__all__ = ['reinplace']

class _ViewType(Enum):
    NonView = 0
    SingleOutputView = 1
    MultiOutputView = 2

class _FunctionalizationMetadataProp(torch.fx.Interpreter):
    def run_node(self, node: Node): ...
    multi_output_view_nodes: Incomplete
    node_counter: int
    def propagate(self, *args): ...

def reinplace(gm, *sample_args):
    '''
    Given an fx.GraphModule, modifies it to perform "reinplacing",
    mutating the nodes of the graph.
    We look for out-of-place op call sites like `b = a.add(...)`,
    and convert them to be inplace (`b = a.add_(...)`),
    as long as the input to the current operator ("a") isn\'t re-used
    anywhere later in the graph.

    This pass currently expects to operate on a **functional, ATen** graph.
    This can be obtained by running `make_fx(functionalize(f))`.

    Sample inputs are needed to determine aliasing relationships of the inputs.
    In general, we can\'t reinplace node `b = a.add(...)` if "a" aliases any of the
    inputs to the program.

    Given a node "b = foo(a, args...) the algorithm for re-inplacing is as follows:

    (1) Perform some initial checks on the metadata of "a" and "args..."
        that can disqualify them from being reinplaced.

      (1a) Check that the self argument we\'re attempting to reinplace
           has acceptable dtype/size metadata to reinplace with.

           For example, if we have:
             a = torch.ones(1)
             b = torch.ones(10)
             out = torch.add(a, b)
           We can\'t turn that into
             a.add_(b)
           Because that would require resizing "a".

           Similarly, we can\'t convert torch.ge(a, b) into a.ge_(b),
           because that would require changing a\'s dtype (from e.g. float32 to bool).
           Note that in this specific example, we could technically do better..

           If we see the pattern:
             a_1 = a.ge(b)
             a_2 = aten._to_copy(a_1, a.dtype)
           Then we this should be valid to completely re-inplace
           (this is exactly what functionalization will emit when it sees a.ge_(b)).

           This optimization is only really important for user programs
           that directly use inplace comparison ops though.

           We also cannot re-inplace on tensors that have overlapping memory,
           e.g. torch.ones(1).expand(4, 4).add_(1)

      (1b) Check if "a" is an alias of any of the program inputs.

          If it is, skip and move to the next node.
          Inplace\'ing an op that would cause it to mutate a program is not sound,
          because that would be a side effect visible to the user.

          NOTE: there\'s a future optimization that we should make:
          if "a" is a (alias of a)  program input, but later in the program
          there is a node that looks like "a.copy_(...)",
          Then re-inplacing is ok to do - we are temporarily re-using a\'s buffer,
          which will later be overwritten by the copy_() call.

          This will be an important optimization to have for programs that mutate
          their inputs. It currently isn\'t implemented though.

      (1c) Check if "a" and "args..." alias

          For example, re-inplacing to create code like the below
          isn\'t guaranteed to be sound:

            aten.mul_(a, a)

    (2) Check that "a" and all of its outstanding aliases are not used anywhere
        later in the graph. If this is the case, then it\'s safe to re-inplace
        to "b = foo_(a)".

        There are a few caveats to this, explained in more detail below:
        (a) If "a" is used later as an argument to a view op, that is okay.
            It\'s only a problem if "a" (or that view) is later passed
            into a normal operator, or if it is returned as the program output.
        (b) If "a" is a repeat argument in `foo()`, then don\'t reinplace.
            Most ATen kernels don\'t make any guarantees that this is sound,
            e.g. if you do aten.mul_(a, a).
            So we\'ll just ban re-inplacing in this case.
            It\'s only a problem if "a" (or that view) is later passed
        (c) If "a" is used as an input into a view "inverse" / "scatter"
            operator, it is potentially fine to re-inplace
            (and remove that scatter operator from the graph).
            See below for a more detailed example.

        NOTE: there is an optimization in this step that is crucial
        to fully recovering performance from functionalization.

        Given this program:
        def f(x):
            a = torch.ops.aten.add(x, x)
            b = torch.ops.aten.diagonal(a)
            torch.ops.aten.fill_(b, 0)
            return d

        Functionalization will emit the following:
        def f(x):
            a = torch.ops.aten.add(x, x)
            b = torch.ops.aten.diagonal(a, 0, 1)
            b_updated = torch.ops.aten.fill(b, 0)
            a_updated = torch.ops.aten.diagonal_scatter(a, b_updated, 0, 1)
            return a_updated

        Ordinarily, we would not be able to reinplace the fill,
        because "b" aliases with "a" which is used by the diagonal_scatter call.

        "re-inplacing" is on the hook for figuring out that it is ok to
        completely, the expensive diagonal_scatter call, if we re-inplace the add().

        So, for every `alias in alias_set(a)`, instead of checking
        that "alias" is not used anywhere later in the graph,
        we check that
            EITHER:
          (a) alias is not used anywhere later in the graph
            OR:
          (b) alias is used exactly once later on in the graph,
              in the following op:

                out = foo_scatter(alias, x, args...)

              where the following must hold:
                (i) "foo_scatter" is the "inverse" operator for foo.
                    This only applies to "foo" ops that are view operators,
                    which view into a subset of the original tensor\'s memory.
                    In practice, there are ~4 operators where this applies:
                      diagonal -> diagonal_scatter
                      slice -> slice_scatter
                      select -> select_scatter
                      as_strided -> as_strided_scatter
                (ii) "args..." are the same between the foo() and foo_scatter() calls.

    (3) Perform the actual re-inplacing on foo!

      (3b) is the common case, but special care is needed for {view}_scatter (3a)

      (3a) {view}_scatter ops.

        Consider this program:
          a = torch.zeros(2, 2)
          b = torch.ones(2)
          a[0] = b

        Post functionalization, that will look like:
          a = torch.zeros(2)
          b = torch.ones(1)
          a_updated = torch.select_scatter(a, b, 0, 0)

        In this case though, there is no "functional" op to re-inplace!
        Instead, we\'d like to directly remove toe select_scatter call.
        We already know from (3) that this is valid,
        because "a" has no later usages in the graph.

        We perform the re-inplacing on the {view}_scatter op like so
        Before:
          a_updated = torch.select_scatter(a, b, args...)
        After:
          a_slice = a.select(a, args...)
          a_slice.copy_(b)

      (3b) Otherwise, replace the functional op with its inplace variant.
        Before:
          b = foo(a, args...)
        After:
          a.foo_(args...)

    (4) Finally, after converting either:
          Before:
            b = foo(a)
          After:
            foo_(a)
        or
          Before:
            b = {slice}_scatter(a, mutated_slice, args...)
          After:
            slice = {slice}(a, args...)
            slice.copy_(mutated_slice)

        We now need to find all later nodes that use "b" as an argument
        and update them to take in "a" instead.

        Note that for the majority of inplace ops, this isn\'t actually necessary
        (because most inplace ops return "self" as their output).
        This isn\'t generally true for all mutable ops though, which is why
        we need to actually replace all of the arguments.

        We also need to update our metadata of Dict[StorageWeakRef, Set[Node]],
        That maps a given tensor storage to the set of all nodes that take in that storage
        as an input.
        Specifically, re-inplacing `b = foo(a)` causes "a" and "b"\'s sets to get fused
        together.

    (5) Any "view_inverse/scatter" nodes that were identified as "it\'s ok to ignore them"
        during step (3) get manually deleted from the graph.
        Their outputs are no longer used, so technically standard DCE would be able
        to do this, but we can no longer run FX\'s DCE pass now that we have mutable
        ops in the graph.
    '''
