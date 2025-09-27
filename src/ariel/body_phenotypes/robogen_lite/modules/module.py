"""TODO(jmdm): description of script.

Todo:
----
    [ ] Autoincrement of index in subclasses.
"""

# Standard library
import typing
from abc import ABC, abstractmethod


class Module(ABC):
    """Base class for all modules."""

    required_attributes: typing.ClassVar[list[str]] = ["index", "module_type"]

    def __init_subclass__(cls) -> None:
        """
        Ensure that subclasses define required attributes.

        Raises
        ------
        NotImplementedError
            If a required attribute is not defined in the subclass.
        """
        super().__init_subclass__()
        for attr in cls.required_attributes:
            if not hasattr(cls, attr):
                msg = f"Class '{cls.__name__}' must define attribute '{attr}'"
                raise NotImplementedError(msg)

    @abstractmethod
    def rotate(self, angle: float) -> None:
        """
        Rotate the module by a certain angle.

        Parameters
        ----------
        angle : float
            The angle to rotate the module by, in degrees.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        msg = f"{self.__class__.__name__} does not implement 'rotate' method."
        raise NotImplementedError(msg)
