import torch
from ..pattern_matcher import Arg as Arg, CallFunction as CallFunction, CallFunctionVarArgs as CallFunctionVarArgs, CallMethodVarArgs as CallMethodVarArgs, FailedMatch as FailedMatch, Ignored as Ignored, KeywordArg as KeywordArg, ListOf as ListOf, MULTIPLE as MULTIPLE, Match as Match, MatchContext as MatchContext, PatternExpr as PatternExpr, PatternMatcherPass as PatternMatcherPass, RepeatedExpr as RepeatedExpr, get_arg_value as get_arg_value, register_graph_pattern as register_graph_pattern
from .group_batch_fusion import POST_GRAD_FUSIONS as POST_GRAD_FUSIONS, PRE_GRAD_FUSIONS as PRE_GRAD_FUSIONS, is_node_meta_valid as is_node_meta_valid
from _typeshed import Incomplete
from torch._dynamo.utils import counters as counters
from torch.fx.experimental.symbolic_shapes import free_symbols as free_symbols
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable
from typing_extensions import TypeAlias

log: Incomplete
_Arguments: TypeAlias
_TransformParam: TypeAlias = tuple[_Arguments | None, _Arguments | None, _Arguments | None, _Arguments | None]
_Range: TypeAlias = tuple[int, int]
PRE_GRAD_PATTERNS: dict[str, PatternMatcherPass]
POST_GRAD_PATTERNS: dict[str, PatternMatcherPass]
pre_grad_pass_names: Incomplete
post_grad_pass_names: Incomplete

def construct_pattern_matcher_pass(pass_name: str):
    """
    Return the specific pattern_matcher_pass given the pass name.
    """
def _get_split_args_default(split_node): ...
def _get_dim(node: Any): ...
def normalize_split_base(match: Match, _get_split_args: Callable[[torch.fx.Node], tuple[torch.fx.Node | None, Any | None, int | None]]):
    """
    Normalize split with split_size into split_with_sizes, so that we only deal with one type of split in
    subsequent optimizations
    """
def normalize_split_default(match: Match, *args, **kwargs): ...
def remove_split_with_size_one(match: Match, *args, **kwargs): ...
def normalize_unbind_default(match: Match, *args, **kwargs): ...
def normalize_cat_default(match: Match, *args, **kwargs): ...
def normalize_stack_default(match: Match, *args, **kwargs): ...
def find_next_users(split_node: torch.fx.Node) -> list[torch.fx.Node]: ...
def normalize_squeeze_default(match: Match, *args, **kwargs): ...
def normalize_reshape_default(match: Match, *args, **kwargs): ...
def normalize_clamp_default(match: Match, *args, **kwargs): ...
def normalize_detach_default(match: Match, *args, **kwargs): ...

class TorchSplit(CallFunction):
    """
    Matches a call to torch.split if it is in a normalized form. Ensures that all users of
    splits are unique getitems.
    """
    def __init__(self, arg, sizes, func=...) -> None: ...
    def _match(self, node: torch.fx.Node, ctx: MatchContext): ...

def merge_splits(match: Match, first_split_input: torch.fx.Node, first_split_sections: list[int], next_split_sections: list[int], dim: int): ...

class SplitCatSimplifier:
    '''
    Helper class to simplify split-cat pattern. In simple cases, both split and cat node can be removed in a "split->cat"
    pattern. However, there are various cases where they can\'t and we need to simplify split/ add transforms before cat.
    Some such cases are:
        1. Final node has additional args (not coming from the initial split)
        2. Shuffling of args between split/cat
        3. Some final nodes are non-(cat/stack)
        4. Split-dim != cat-dim (but equal split)

    Note that any combination of the above cases can happen.

    To deal with 1, 2, & 3 - we iterate over all users of split. And figure out common "ranges" that can be merged.
    Then, we simplify the split accordingly. In the best case, split can be entirely removed.

    To deal with 4, we add some transformations (unflatten + movedim) (See `get_transform_params`).

    Finally, depending on final node being cat or stack, unsqueeze/flatten needs to be added.

    '''
    def simplify(self, graph: torch.fx.Graph, split_node: torch.fx.Node, split_sections: list[int]): ...
    def get_user_input_list(self, split_node: torch.fx.Node, next_users: list[torch.fx.Node]) -> list[list[torch.fx.Node | _Range]]:
        '''
        Returns list of inputs to the following user nodes, in order. The outer list represents the user node. The inner
        list represents the inputs to that particular node. This list can either contain
          - a tuple representing the ranges of get_items that should go into the cat (closed interval)
          - torch.fx.Node representing "other" inputs (which are not coming from our split)
        '''
    def get_merged_user_inputs(self, split_node: torch.fx.Node, cat_node: torch.fx.Node) -> list[torch.fx.Node | _Range]: ...
    def get_non_cat_node_input(self, split_node: torch.fx.Node, node: torch.fx.Node) -> list[_Range]:
        """
        Get input for a non cat node in the same format as `get_merged_user_inputs`
        """
    def merge_consecutive_inputs(self, inputs: list[torch.fx.Node | int]) -> list[torch.fx.Node | _Range]:
        """
        Merge consecutive inputs going into a user node.

        For e.g.
        [arg0, 0, 1, 2, arg1] -> [arg0, (0, 2), arg1]
        """
    def get_simplified_split_ranges(self, split_sections, next_users, user_inputs_list: list[list[torch.fx.Node | _Range]]) -> list[_Range] | None: ...
    def has_non_overlapping_ranges(self, ranges: list[_Range]) -> bool: ...
    def fill_gaps(self, ranges: list[_Range], min_: int, max_: int) -> list[_Range]: ...
    def get_transform_params(self, split_node: torch.fx.Node, next_users: list[torch.fx.Node], user_inputs_list: list[list[torch.fx.Node | _Range]]) -> list[list[_TransformParam]] | None:
        """
        Figure out what transforms are needed for each input to each cat node.

        We replace a split node with an unflatten followed by a movedim
        """
    def replace_split(self, graph: torch.fx.Graph, split_node: torch.fx.Node, split_sections: list[int], user_inputs_list: list[list[torch.fx.Node | _Range]], split_ranges: list[_Range]) -> list[list[torch.fx.Node]]:
        """
        Replace the split node. It can either remove the split node if len(split_ranges) == 1, or simplify it
        into a split with lesser sections if len(split_ranges) > 1.

        Returns the new `user_inputs_list`, with tuples replaced with new getitems from the newer split node.
        """
    def replace_cat(self, graph: torch.fx.Graph, split_node: torch.fx.Node, next_users: list[torch.fx.Node], user_inputs_list_new, transform_params_list: list[list[_TransformParam]]): ...
    def erase_old_nodes(self, graph: torch.fx.Graph, split_node: torch.fx.Node, next_users: list[torch.fx.Node]): ...

class UnbindCatRemover(SplitCatSimplifier):
    """
    Helper class to merge Unbind->Cat/Stack. Many of the cases are similar to SplitCatSimplifier.

    Unbind can't be simplified like splits. So, we can only remove the unbind node. Other than this,
    other cases like multiple users, additional args, dim mismatch are similar to `SplitCatSimplifier`,
    hence we extend that class.
    """
    def remove_unbind(self, graph: torch.fx.Graph, unbind_node: torch.fx.Node): ...
    def get_simplified_split_ranges(self, split_sections: list[int], next_users: list[torch.fx.Node], user_inputs_list: list[list[torch.fx.Node | _Range]]) -> list[_Range] | None: ...
    def get_transform_params(self, split_node: torch.fx.Node, next_users: list[torch.fx.Node], user_inputs_list: list[list[torch.fx.Node | _Range]]) -> list[list[_TransformParam]] | None:
        """
        Figure out what transforms are needed for each input to each cat node.

        Here is the rough transforms we apply:

        x -> unbind -> stack => x -> movedim

        x -> unbind -> cat => x -> movedim -> flatten

        When cat/stack nodes have additional args:

             addn ---|              addn -> unsqueeze ---|
        x -> unbind -> stack  =>           x -> movedim  -> cat

             addn ---|                            addn ---|
        x -> unbind -> cat  =>   x -> movedim -> flatten  -> cat

        (Note application of these depends on the dims as well)


        """

class GetItem(CallFunction):
    def __init__(self, arg, index, _users: int = 1) -> None: ...
    def find_anchor_nodes(self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]): ...

def merge_split_squeeze(match: Match, split_input: torch.fx.Node, split_sizes: list[int], dim: int): ...

getitem_unbind: Incomplete

def merge_unbind_stack(match: Match, unbind_input: torch.fx.Node, dim: int): ...

getitem_split: Incomplete
reshape_getitem_split: Incomplete

def simplify_split_cat(match: Match, split_sections: list[int], dim: int): ...
def has_same_parent_node(node: torch.fx.Node): ...
def remove_zeros(split_sections: list[int]):
    """
    Remove zeros from the list and get the index mapping dict from getitem
    in split node to getitem in new split node
    """
def is_sorted_and_consecutive(arr: list[int]) -> bool: ...
def calculate_fused_tensor_size(split_node: torch.fx.Node, indices: list[int]) -> int:
    """
    Calculate the fused tensor size in the indices
    """
def merge_getitem_cat(match: Match, split_sections: list[int], dim: int): ...
def mutate_cat_node(match: Match, split_sections: list[int], dim: int): ...

getitem_split_aten: Incomplete

def normalize_split_default_aten(match: Match, *args, **kwargs): ...
def normalize_split_with_size_default_aten(match: Match, *args, **kwargs): ...
def merge_split_cat_aten(match: Match, *args, **kwargs): ...
def merge_select_cat_aten(match: Match, *args, **kwargs): ...
def normalize_cat_default_aten(match: Match, *args, **kwargs): ...
def merge_unbind_stack_aten(match: Match, *args, **kwargs): ...
def divide_into_consecutive_sublists(indices: list[int]) -> list[list[int]]: ...
def update_args_from_split_getitem(graph: torch.fx.Graph, node: torch.fx.Node, getitem_indices: list[int], parents_seen: list[torch.fx.Node], new_cat_args: list[torch.fx.Node], new_cat_args_meta: list[torch.fx.Node], idx_to_getitems: dict[int, torch.fx.Node], threshold_to_cat: int = 2): ...
def reshape_cat_node(graph: torch.fx.Graph, cat_node: torch.fx.Node, unbind_input: torch.fx.Node, cat_dim: int, unbind_dim: int, cat_shape: torch.Size) -> torch.fx.Node: ...
def update_args_from_unbind_getitem(graph: torch.fx.Graph, node: torch.fx.Node, getitem_indices: list[int], parents_seen: list[torch.fx.Node], new_cat_args: list[torch.fx.Node], new_cat_args_meta: list[torch.fx.Node], idx_to_getitems: dict[int, torch.fx.Node], threshold_to_cat: int = 2): ...
def construct_cat_args(graph: torch.fx.Graph, cat_or_stack_node: torch.fx.Node, inputs: list[torch.fx.Node], split_or_unbind_node: torch.fx.Node, threshold_to_cat: int = 2, run_update_func: Callable = ...) -> tuple[list[torch.fx.Node], list[torch.Tensor]]: ...
def remove_split_unbind_children(graph: torch.fx.Graph, inputs: list[torch.fx.Node]): ...
def split_cat_to_slices(match: Match, split_sections: list[int], dim: int): ...
def unbind_cat_to_view(match: Match, unbind_input: torch.fx.Node, dim: int): ...
def reshape_cat_node_to_stack(graph: torch.fx.Graph, cat_node: torch.fx.Node, stack_node: torch.fx.Node, split_or_unbind_dim: int) -> None: ...
def convert_reshape_cat_arg_to_stack(graph: torch.fx.Graph, cat_node: torch.fx.Node, stack_node: torch.fx.Node, stack_node_shape: torch.Size, stack_dim: int, split_dim: int) -> torch.fx.Node: ...
def split_stack_to_cats(match: Match, split_sections: list[int], dim: int): ...
def unbind_stack_to_slices(match: Match, unbind_input: torch.fx.Node, dim: int): ...
def get_view_shape_list(cat_arg: torch.fx.Node, stack_dim: int) -> list[int]: ...
def move_reshape_out_of_split_stack(match: Match, *args, **kwargs): ...

view_getitem_split_aten: Incomplete

def move_view_after_cat(match: Match, *args, **kwargs): ...
