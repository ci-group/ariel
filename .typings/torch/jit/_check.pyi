import ast
import torch
from _typeshed import Incomplete

class AttributeTypeIsSupportedChecker(ast.NodeVisitor):
    '''Check the ``__init__`` method of a given ``nn.Module``.

    It ensures that all instance-level attributes can be properly initialized.

    Specifically, we do type inference based on attribute values...even
    if the attribute in question has already been typed using
    Python3-style annotations or ``torch.jit.annotate``. This means that
    setting an instance-level attribute to ``[]`` (for ``List``),
    ``{}`` for ``Dict``), or ``None`` (for ``Optional``) isn\'t enough
    information for us to properly initialize that attribute.

    An object of this class can walk a given ``nn.Module``\'s AST and
    determine if it meets our requirements or not.

    Known limitations
    1. We can only check the AST nodes for certain constructs; we can\'t
    ``eval`` arbitrary expressions. This means that function calls,
    class instantiations, and complex expressions that resolve to one of
    the "empty" values specified above will NOT be flagged as
    problematic.
    2. We match on string literals, so if the user decides to use a
    non-standard import (e.g. `from typing import List as foo`), we
    won\'t catch it.

    Example:
        .. code-block:: python

            class M(torch.nn.Module):
                def fn(self):
                    return []

                def __init__(self) -> None:
                    super().__init__()
                    self.x: List[int] = []

                def forward(self, x: List[int]):
                    self.x = x
                    return 1

        The above code will pass the ``AttributeTypeIsSupportedChecker``
        check since we have a function call in ``__init__``. However,
        it will still fail later with the ``RuntimeError`` "Tried to set
        nonexistent attribute: x. Did you forget to initialize it in
        __init__()?".

    Args:
        nn_module - The instance of ``torch.nn.Module`` whose
            ``__init__`` method we wish to check
    '''
    class_level_annotations: Incomplete
    visiting_class_level_ann: bool
    def check(self, nn_module: torch.nn.Module) -> None: ...
    def _is_empty_container(self, node: ast.AST, ann_type: str) -> bool: ...
    def visit_Assign(self, node) -> None:
        """Store assignment state when assigning to a Call Node.

        If we're visiting a Call Node (the right-hand side of an
        assignment statement), we won't be able to check the variable
        that we're assigning to (the left-hand side of an assignment).
        Because of this, we need to store this state in visitAssign.
        (Luckily, we only have to do this if we're assigning to a Call
        Node, i.e. ``torch.jit.annotate``. If we're using normal Python
        annotations, we'll be visiting an AnnAssign Node, which has its
        target built in.)
        """
    def visit_AnnAssign(self, node) -> None:
        """Visit an AnnAssign node in an ``nn.Module``'s ``__init__`` method.

        It checks if it conforms to our attribute annotation rules."""
    def visit_Call(self, node) -> None:
        """Determine if a Call node is 'torch.jit.annotate' in __init__.

        Visit a Call node in an ``nn.Module``'s ``__init__``
        method and determine if it's ``torch.jit.annotate``. If so,
        see if it conforms to our attribute annotation rules.
        """
