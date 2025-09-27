from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from typing import Callable

__all__ = ['QConfigMultiMapping']

class QConfigMultiMapping:
    '''
    This class, used with the prepare_n_shadows_model API, stores a list of :class:`torch.ao.quantization.QConfigMapping`s
    so that multiple QConfigs can be specified for each QConfig matching style.

    The user can specify QConfigs using the following methods (in increasing match priority):

        ``set_global`` : sets the global (default) QConfigs

        ``set_object_type`` : sets the QConfigs for a given module type, function, or method name

        ``set_module_name_regex`` : sets the QConfigs for modules matching the given regex string

        ``set_module_name`` : sets the QConfigs for modules matching the given module name

        ``set_module_name_object_type_order`` : sets the QConfigs for modules matching a combination
        of the given module name, object type, and the index at which the module appears

    Note: Usage of set methods is the same as in QConfigMapping except with a passed in list of QConfigs rather than a
    single QConfig.

    Example usage::

        qconfig_mapping = QConfigMultiMapping()
            .set_global([qconfig1, qconfig2])
            .set_object_type(torch.nn.Linear, [qconfig2, qconfig3])
            .set_object_type(torch.nn.ReLU, [qconfig1])
            .set_module_name_regex("foo.*bar.*conv[0-9]+", [qconfig2])
            .set_module_name_regex("foo.*", [qconfig1, qconfig2, qconfig3])
            .set_module_name("module1", [None])
            .set_module_name("module2", [qconfig2])
            .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, [qconfig3])

    '''
    qconfig_mappings_list: list[QConfigMapping]
    def __init__(self) -> None: ...
    def _handle_list_size_mismatch(self, qconfig_list: list[QConfigAny], style: str) -> None: ...
    def _insert_qconfig_list(self, style: str, args: list[str | int | Callable], qconfig_list: list[QConfigAny]) -> None: ...
    def set_global(self, global_qconfig_list: list[QConfigAny]) -> QConfigMultiMapping:
        """
        Set global QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_global()` for more info
        """
    def set_object_type(self, object_type: Callable | str, qconfig_list: list[QConfigAny]) -> QConfigMultiMapping:
        """
        Set object type QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_object_type()` for more info
        """
    def set_module_name_regex(self, module_name_regex: str, qconfig_list: list[QConfigAny]) -> QConfigMultiMapping:
        """
        Set module_name_regex QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name_regex()` for more info
        """
    def set_module_name(self, module_name: str, qconfig_list: list[QConfigAny]) -> QConfigMultiMapping:
        """
        Set module_name QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name()` for more info
        """
    def set_module_name_object_type_order(self, module_name: str, object_type: Callable, index: int, qconfig_list: list[QConfigAny]) -> QConfigMultiMapping:
        """
        Set module_name QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name_object_type_order()` for more info
        """
    def __repr__(self) -> str: ...
    @classmethod
    def from_list_qconfig_mapping(cls, qconfig_mapping_list: list[QConfigMapping]) -> QConfigMultiMapping:
        """
        Creates a QConfigMultiMapping from a list of QConfigMappings
        """
