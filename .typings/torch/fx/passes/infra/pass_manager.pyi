import torch.nn as nn
from _typeshed import Incomplete
from torch.fx.passes.infra.pass_base import PassResult
from typing import Callable

__all__ = ['pass_result_wrapper', 'this_before_that_pass_constraint', 'PassManager']

def pass_result_wrapper(fn: Callable) -> Callable:
    '''
    Wrapper for passes which currently do not return a PassResult.
    This wrapper makes them return a PassResult containing the modified object
    and True for the "modified" flag.

    Args:
        fn (Callable[Module, Any])

    Returns:
        wrapped_fn (Callable[Module, PassResult])
    '''
def this_before_that_pass_constraint(this: Callable, that: Callable) -> Callable:
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.

    For example, the following pass list and constraint list would be invalid.
    ```
    passes = [pass_b, pass_a]

    constraints = [this_before_that_pass_constraint(pass_a, pass_b)]
    ```

    Args:
        this (Callable): pass which should occur first
        that (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool]
    """

class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): List of passes. A pass is a
            callable which modifies an object and returns a PassResult
        constraint (Optional[List[Callable]]): List of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
        steps (int): Max number of times we run the passes (default = 1).
        run_checks_after_each_pass (bool): Whether to run checks and linting
            after each pass
        suppress_check_failures (bool): Whether to raise errors when running
            checks
    """
    passes: list[Callable[[nn.Module], PassResult]]
    constraints: list[Callable[[Callable, Callable], bool]]
    _validated: bool
    steps: int
    run_checks_after_each_pass: Incomplete
    suppress_check_failures: Incomplete
    def __init__(self, passes=None, constraints=None, steps=None, run_checks_after_each_pass: bool = False, suppress_check_failures: bool = False) -> None: ...
    def add_pass(self, _pass: Callable):
        """
        Adds a pass into the current list of passes.
        """
    def add_constraint(self, constraint: Callable):
        """
        Adds a constraint into the current list of constraints.
        """
    def validate_constraints(self) -> None:
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
    def solve_constraints(self) -> None:
        """
        Finds a valid traversal order based on the given constraints and orders
        the passes based on this order.

        If a circular dependency exists between the constraints and steps = 1,
        then we will raise an error because if steps != 1 this means that we
        will re-run the passes, allowing for circular dependencies.
        """
    def add_checks(self, check: Callable) -> None:
        """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
    def check(self, module: nn.Module) -> None: ...
    def __call__(self, module: nn.Module) -> PassResult:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module if `run_checks_after_each_pass` is set.

        If the module is a graph module, we will run the list of passes until
        the graph stops changing, or until `steps` number of times.
        """
