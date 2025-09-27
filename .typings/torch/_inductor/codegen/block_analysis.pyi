from ..utils import sympy_dot as sympy_dot, sympy_subs as sympy_subs
from ..virtualized import V as V
from sympy import Expr as Expr, Symbol as Symbol
from torch.utils._sympy.functions import FloorDiv as FloorDiv, ModularIndexing as ModularIndexing

class BlockPatternMatcher:
    """
    Matches block indexing expressions.
    """
    @classmethod
    def get_subexpr_involving_symbol(cls, expr: Expr, symbol: Symbol) -> Expr:
        """
        Given a sympy expression, return the subexpression comprised only of terms
        involving the specified symbol.

        For example, if `expr` is `x * 5 + x ** 2 + y * 2 + 5`, and `symbol` is `x`,
        this returns `x * 5 + x ** 2`.
        """
    @staticmethod
    def get_slice_numels(dims: list[Expr]) -> list[Expr]:
        """
        Compute the cumulative size of each dimension's slice.
        This proceeds from the last dim up to the second.
        """
    @staticmethod
    def _preprocess(expr: Expr) -> Expr: ...
    @classmethod
    def match_mod_div_block_expr(cls, index: Expr, index_var: Symbol, numel: Expr, num_dims: int) -> tuple[list[Expr], list[Expr], list[Expr]] | None:
        """
        Matches modular indexing expressions, converting them to implied block dimensions and strides.
        See triton.py for more information.
        """
    @classmethod
    def match_affine_block_expr(cls, index: Expr, index_var: Symbol) -> Expr | None:
        """
        Matches simple expressions of the form stride * index, returning the
        stride.
        """
