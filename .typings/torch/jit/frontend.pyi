from _typeshed import Incomplete
from torch import _jit_internal as _jit_internal
from torch._C._jit_tree_views import Apply as Apply, Assert as Assert, Assign as Assign, Attribute as Attribute, AugAssign as AugAssign, BinOp as BinOp, Break as Break, ClassDef as ClassDef, Const as Const, Continue as Continue, Decl as Decl, Def as Def, Delete as Delete, DictComp as DictComp, DictLiteral as DictLiteral, Dots as Dots, EmptyTypeAnnotation as EmptyTypeAnnotation, ExprStmt as ExprStmt, FalseLiteral as FalseLiteral, For as For, Ident as Ident, If as If, ListComp as ListComp, ListLiteral as ListLiteral, NoneLiteral as NoneLiteral, Param as Param, Pass as Pass, Property as Property, Raise as Raise, Return as Return, Select as Select, SliceExpr as SliceExpr, Starred as Starred, Stmt as Stmt, StringLiteral as StringLiteral, Subscript as Subscript, TernaryIf as TernaryIf, TrueLiteral as TrueLiteral, TupleLiteral as TupleLiteral, UnaryOp as UnaryOp, Var as Var, While as While, With as With, WithItem as WithItem
from torch._jit_internal import FunctionModifiers as FunctionModifiers, _is_drop_fn as _is_drop_fn, is_static_fn as is_static_fn, should_drop as should_drop
from torch._sources import get_source_lines_and_file as get_source_lines_and_file, make_source_context as make_source_context, parse_def as parse_def
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS as DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name as get_qualified_name, monkeytype_trace as monkeytype_trace

_IS_ASTUNPARSE_INSTALLED: bool
_reserved_prefix: str
_reserved_names: Incomplete
_identifier_chars: Incomplete

def is_reserved_name(name): ...

pretty_node_names: Incomplete
node_start_tokens: Incomplete

class FrontendError(Exception):
    source_range: Incomplete
    msg: Incomplete
    error_report: Incomplete
    def __init__(self, source_range, msg) -> None: ...
    def __str__(self) -> str: ...

class NotSupportedError(FrontendError): ...

class UnsupportedNodeError(NotSupportedError):
    def __init__(self, ctx, offending_node, reason: str = '') -> None: ...

class FrontendTypeError(FrontendError): ...

def build_withitems(ctx, items): ...
def build_stmts(ctx, stmts): ...
def get_class_properties(cls, self_name):
    """
    Get a list of Property objects representing the properties of a class.

    Args:
        cls:  The class to get properties of.
        self_name: The name of the class that the properties should belong to.
    Returns:
        A list of Property objects corresponding to the properties of cls. Property
        here refers to the subclass of TreeView.
    """
def get_class_assigns(ctx, cls_ast): ...
def get_jit_class_def(cls, self_name):
    """Get definitions for each method within the current class independently.

    Args:
        cls: The class to get definition of.
        self_name: The name of the class that the properties should belong to.

    Returns:
        torch._C._jit_tree_views.ClassDef: A representation of the class,
            the methods in the class and their definition as a tree.
    """
def get_jit_def(fn, def_name, self_name=None, is_classmethod: bool = False):
    '''
    Build a JIT AST (TreeView) from the given function.

    Args:
        fn: A function object to compile or a pre-parsed ParsedDef object
        def_name: The name to give to the resulting AST object. This is not
            always the same as `fn.__name__`, for example:
                def _forward(self):
                    ...
                forward = _forward
            In this case, the `__name__` attribute of the function object is "_forward",
            but we want the result AST to have the name "forward".
        self_name: If this function is a method, what the type name of `self` is.
    '''
def is_torch_jit_ignore_context_manager(stmt): ...

class Builder:
    def __call__(self, ctx, node): ...

def build_class_def(ctx, py_def, methods, properties, self_name, assigns): ...
def build_def(ctx, py_def, type_line, def_name, self_name=None, pdt_arg_types=None): ...

_vararg_kwarg_err: str

def build_param_list(ctx, py_args, self_name, pdt_arg_types=None): ...
def build_param(ctx, py_arg, self_name, kwarg_only, pdt_arg_type=None): ...
def build_ignore_context_manager(ctx, stmt): ...
def get_default_args(fn):
    """
    Get a dictionary of default arguments for a function.

    Args:
        fn: Callable - The function to inspect for default arguments.
    Returns:
        (Dict[str, Any]): mapping argument names to their default values if
        :attr:`fn` is not None, else empty dictionary.
    """
def get_default_args_for_class(cls):
    """
    Get default arguments for all methods in a class (except for static methods).

    Args:
        cls: type - The class type to inspect for default arguments.
    Returns:
        A Dict[str, Dict[str, Any]] which maps each method name to a Dict[str, Any]
        that maps each argument name to its default value.
    """

class WithItemBuilder(Builder):
    @staticmethod
    def build_withitem(ctx, item): ...

class StmtBuilder(Builder):
    augassign_map: Incomplete
    @staticmethod
    def build_Expr(ctx, stmt): ...
    @staticmethod
    def build_Assign(ctx, stmt): ...
    @staticmethod
    def build_AnnAssign(ctx, stmt): ...
    @staticmethod
    def build_Delete(ctx, stmt): ...
    @staticmethod
    def build_Return(ctx, stmt): ...
    @staticmethod
    def build_Raise(ctx, stmt): ...
    @staticmethod
    def build_Assert(ctx, stmt): ...
    @staticmethod
    def build_AugAssign(ctx, stmt): ...
    @staticmethod
    def build_While(ctx, stmt): ...
    @staticmethod
    def build_For(ctx, stmt): ...
    @staticmethod
    def build_If(ctx, stmt): ...
    @staticmethod
    def build_Print(ctx, stmt): ...
    @staticmethod
    def build_Pass(ctx, stmt): ...
    @staticmethod
    def build_Break(ctx, stmt): ...
    @staticmethod
    def build_Continue(ctx, stmt): ...
    @staticmethod
    def build_With(ctx, stmt): ...

class ExprBuilder(Builder):
    binop_map: Incomplete
    unop_map: Incomplete
    boolop_map: Incomplete
    cmpop_map: Incomplete
    @staticmethod
    def build_Attribute(ctx, expr): ...
    @staticmethod
    def build_Call(ctx, expr): ...
    @staticmethod
    def build_Ellipsis(ctx, expr): ...
    @staticmethod
    def build_Name(ctx, expr): ...
    @staticmethod
    def build_NameConstant(ctx, expr): ...
    @staticmethod
    def build_BinOp(ctx, expr): ...
    @staticmethod
    def build_UnaryOp(ctx, expr): ...
    @staticmethod
    def build_BoolOp(ctx, expr): ...
    @staticmethod
    def build_IfExp(ctx, expr): ...
    @staticmethod
    def build_Compare(ctx, expr): ...
    @staticmethod
    def build_Subscript(ctx, expr): ...
    @staticmethod
    def build_List(ctx, expr): ...
    @staticmethod
    def build_Tuple(ctx, expr): ...
    @staticmethod
    def build_Dict(ctx, expr): ...
    @staticmethod
    def build_Num(ctx, expr): ...
    @staticmethod
    def build_Constant(ctx, expr): ...
    @staticmethod
    def build_Str(ctx, expr): ...
    @staticmethod
    def build_JoinedStr(ctx, expr): ...
    @staticmethod
    def build_ListComp(ctx, stmt): ...
    @staticmethod
    def build_GeneratorExp(ctx, stmt): ...
    @staticmethod
    def build_DictComp(ctx, stmt): ...
    @staticmethod
    def build_Starred(ctx, expr): ...

build_expr: Incomplete
build_stmt: Incomplete
build_withitem: Incomplete

def find_before(ctx, pos, substr, offsets=(0, 0)): ...
