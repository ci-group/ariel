from . import ast_nodes as ast_nodes
from _typeshed import Incomplete
from typing import Mapping, MutableSequence, Sequence

ARRAY_EXTENTS_PATTERN: Incomplete
ARRAY_N_PATTERN: Incomplete
STARTS_WITH_CONST_PATTERN: Incomplete
ENDS_WITH_CONST_PATTERN: Incomplete

def _parse_qualifiers(type_name: str, qualifiers: Sequence[str]) -> tuple[str, Mapping[str, bool]]:
    """Separates qualifiers from the rest of the type name."""
def _parse_maybe_array(type_name: str, innermost_type: ast_nodes.ValueType | ast_nodes.PointerType | None) -> ast_nodes.ValueType | ast_nodes.PointerType | ast_nodes.ArrayType:
    """Internal-only helper that parses a type that may be an array type."""
def _parse_maybe_pointer(type_name: str, innermost_type: ast_nodes.ValueType | ast_nodes.PointerType | None) -> ast_nodes.ValueType | ast_nodes.PointerType | ast_nodes.ArrayType:
    """Internal-only helper that parses a type that may be a pointer type."""
def _peel_nested_parens(input_str: str) -> MutableSequence[str]:
    """Extracts substrings from a string with nested parentheses.

  The returned sequence starts from the substring enclosed in the innermost
  parentheses and moves subsequently outwards. The contents of the inner
  substrings are removed from the outer ones. For example, given the string
  'lorem ipsum(dolor sit (consectetur adipiscing) amet)sed do eiusmod',
  this function produces the sequence
  ['consectetur adipiscing', 'dolor sit  amet', 'lorem ipsumsed do eiusmod'].

  Args:
    input_str: An input_str string consisting of zero or more nested
      parentheses.

  Returns:
    A sequence of substrings enclosed with in respective parentheses. See the
    description above for the precise detail of the output.
  """
def parse_type(type_name: str) -> ast_nodes.ValueType | ast_nodes.PointerType | ast_nodes.ArrayType:
    """Parses a string that represents a C type into an AST node."""
def parse_function_return_type(type_name: str) -> ast_nodes.ValueType | ast_nodes.PointerType | ast_nodes.ArrayType: ...
