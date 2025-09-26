import sympy
from _typeshed import Incomplete
from torch.utils._sympy.functions import FloorDiv as FloorDiv

log: Incomplete
_MIRROR_REL_OP: dict[type[sympy.Basic], type[sympy.Rel]]
INEQUALITY_TYPES: Incomplete

def mirror_rel_op(type: type) -> type[sympy.Rel] | None: ...
def try_solve(expr: sympy.Basic, thing: sympy.Basic, trials: int = 5, floordiv_inequality: bool = True) -> tuple[sympy.Rel, sympy.Expr] | None: ...
def _try_isolate_lhs(e: sympy.Basic, thing: sympy.Basic, floordiv_inequality: bool) -> sympy.Basic: ...
