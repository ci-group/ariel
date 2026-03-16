from __future__ import annotations

from .math_utils import Vector3, Quaternion
from dataclasses import dataclass

@dataclass
class AttachmentPoint:
    """An attachment point on the parent module for the child."""

    orientation: Quaternion
    """The orientation of the attachment point on the module."""
    offset: Vector3
    """The offset for the attachment point, with respect to the center of the module."""


class Module:
    """Base class for a module for modular robots."""

    _attachment_points: dict[int, AttachmentPoint]
    _children: dict[int, Module]

    _parent: Module | None
    """
    The parent module of this module.
    
    None if this module has not yet been added to a body.
    """

    _parent_child_index: int | None
    """
    Index of this module in the parent modules child list.
    
    None if this module has not yet been added to a body.
    """

    def __init__(
        self,
        attachment_points: dict[int, AttachmentPoint],
    ) -> None:
        """
        Initialize this object.
        :param attachment_points: The attachment points available on a module.
        """

        self._attachment_points = attachment_points
        self._children = {}

        self._parent = None
        self._parent_child_index = None

    @property
    def parent(self) -> Module | None:
        """
        Get the parent module of this module.

        None if this module has not yet been added to a body.

        :returns: The parent module of this module, or None if this module has not yet been added to a body.
        """
        return self._parent

    @property
    def parent_child_index(self) -> int | None:
        """
        Index of this module in the parent modules child list.

        None if this module has not yet been added to a body.

        :returns: The index of this module in the parent modules child list, or None if this module has not yet been added to a body.
        """
        return self._parent_child_index

    @property
    def children(self) -> dict[int, Module]:
        """
        Get all children on this module.

        :return: The children and their respective attachment point index.
        """
        return self._children

    def set_child(self, module: Module, child_index: int) -> None:
        """
        Attach a module to a slot.

        :param module: The module to attach.
        :param child_index: The slot to attach it to.
        :raises KeyError: If attachment point is already populated.
        """
        assert (
            module._parent is None
        ), "Child module already connected to a different slot."
        module._parent = self
        module._parent_child_index = child_index
        if self.can_set_child(child_index):
            self._children[child_index] = module
        else:
            raise KeyError("Attachment point already populated")

    def can_set_child(self, child_index: int) -> bool:
        """
        Check if a child can be set on the module.

        :param child_index: The child index.
        :return: The boolean value.
        """
        if self._children.get(child_index, True):
            return True
        return False

    @property
    def attachment_points(self) -> dict[int, AttachmentPoint]:
        """
        Get all attachment points of this module.

        :return: The attachment points.
        """
        return self._attachment_points
    
    @classmethod
    def __find_recur(cls, module: Module, module_type: Module) -> Module:
        modules = []
        if isinstance(module, module_type):
            modules.append(module)
        for child in module.children.values():
            modules.extend(cls.__find_recur(child, module_type))
        return modules

    def find_modules_of_type(self, module_type: Module) -> list[Module]:
        """
        Find all Modules of a certain type in the robot.

        :param module_type: The type.
        :return: The list of Modules.
        """
        return self.__find_recur(self._core, module_type)