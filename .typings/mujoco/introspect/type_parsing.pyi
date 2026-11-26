from . import ast_nodes as ast_nodes
from _typeshed import Incomplete

ARRAY_EXTENTS_PATTERN: Incomplete
ARRAY_N_PATTERN: Incomplete
STARTS_WITH_CONST_PATTERN: Incomplete
ENDS_WITH_CONST_PATTERN: Incomplete

def parse_type(type_name: str) -> ast_nodes.ValueType | ast_nodes.PointerType | ast_nodes.ArrayType:
    """Parses a string that represents a C type into an AST node."""
def parse_function_return_type(type_name: str) -> ast_nodes.ValueType | ast_nodes.PointerType | ast_nodes.ArrayType: ...
