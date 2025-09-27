from typing import Callable

__all__ = ['PassManager', 'inplace_wrapper', 'log_hook', 'loop_pass', 'this_before_that_pass_constraint', 'these_before_those_pass_constraint']

def inplace_wrapper(fn: Callable) -> Callable:
    """
    Convenience wrapper for passes which modify an object inplace. This
    wrapper makes them return the modified object instead.

    Args:
        fn (Callable[Object, Any])

    Returns:
        wrapped_fn (Callable[Object, Object])
    """
def log_hook(fn: Callable, level=...) -> Callable:
    '''
    Logs callable output.

    This is useful for logging output of passes. Note inplace_wrapper replaces
    the pass output with the modified object. If we want to log the original
    output, apply this wrapper before inplace_wrapper.


    ```
    def my_pass(d: Dict) -> bool:
        changed = False
        if "foo" in d:
            d["foo"] = "bar"
            changed = True
        return changed


    pm = PassManager(passes=[inplace_wrapper(log_hook(my_pass))])
    ```

    Args:
        fn (Callable[Type1, Type2])
        level: logging level (e.g. logging.INFO)

    Returns:
        wrapped_fn (Callable[Type1, Type2])
    '''
def loop_pass(base_pass: Callable, n_iter: int | None = None, predicate: Callable | None = None):
    """
    Convenience wrapper for passes which need to be applied multiple times.

    Exactly one of `n_iter`or `predicate` must be specified.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        n_iter (int, optional): number of times to loop pass
        predicate (Callable[Object, bool], optional):

    """
def this_before_that_pass_constraint(this: Callable, that: Callable):
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.
    """
def these_before_those_pass_constraint(these: Callable, those: Callable):
    """
    Defines a partial order ('depends on' function) where `these` must occur
    before `those`. Where the inputs are 'unwrapped' before comparison.

    For example, the following pass list and constraint list would be invalid.
    ```
    passes = [
        loop_pass(pass_b, 3),
        loop_pass(pass_a, 5),
    ]

    constraints = [these_before_those_pass_constraint(pass_a, pass_b)]
    ```

    Args:
        these (Callable): pass which should occur first
        those (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool]
    """

class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): list of passes. A pass is a
            callable which modifies an object and returns modified object
        constraint (Optional[List[Callable]]): list of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
    """
    passes: list[Callable]
    constraints: list[Callable]
    _validated: bool
    def __init__(self, passes=None, constraints=None) -> None: ...
    @classmethod
    def build_from_passlist(cls, passes): ...
    def add_pass(self, _pass: Callable): ...
    def add_constraint(self, constraint) -> None: ...
    def remove_pass(self, _passes: list[str]): ...
    def replace_pass(self, _target, _replacement) -> None: ...
    def validate(self) -> None:
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
    def __call__(self, source): ...
