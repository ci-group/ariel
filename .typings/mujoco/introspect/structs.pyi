from .ast_nodes import AnonymousStructDecl as AnonymousStructDecl, AnonymousUnionDecl as AnonymousUnionDecl, ArrayType as ArrayType, PointerType as PointerType, StructDecl as StructDecl, StructFieldDecl as StructFieldDecl, ValueType as ValueType
from typing import Mapping

STRUCTS: Mapping[str, StructDecl]
