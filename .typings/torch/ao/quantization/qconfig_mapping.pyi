from .qconfig import QConfigAny
from collections import OrderedDict
from typing import Any, Callable

__all__ = ['get_default_qconfig_mapping', 'get_default_qat_qconfig_mapping', 'QConfigMapping']

def get_default_qconfig_mapping(backend: str = 'x86', version: int = 0) -> QConfigMapping:
    '''
    Return the default QConfigMapping for post training quantization.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86" (default), "fbgemm", "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    '''
def get_default_qat_qconfig_mapping(backend: str = 'x86', version: int = 1) -> QConfigMapping:
    '''
    Return the default QConfigMapping for quantization aware training.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86" (default), "fbgemm", "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    '''

class QConfigMapping:
    '''
    Mapping from model ops to :class:`torch.ao.quantization.QConfig` s.

    The user can specify QConfigs using the following methods (in increasing match priority):

        ``set_global`` : sets the global (default) QConfig

        ``set_object_type`` : sets the QConfig for a given module type, function, or method name

        ``set_module_name_regex`` : sets the QConfig for modules matching the given regex string

        ``set_module_name`` : sets the QConfig for modules matching the given module name

        ``set_module_name_object_type_order`` : sets the QConfig for modules matching a combination
        of the given module name, object type, and the index at which the module appears

    Example usage::

        qconfig_mapping = QConfigMapping()
            .set_global(global_qconfig)
            .set_object_type(torch.nn.Linear, qconfig1)
            .set_object_type(torch.nn.ReLU, qconfig1)
            .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
            .set_module_name_regex("foo.*", qconfig2)
            .set_module_name("module1", qconfig1)
            .set_module_name("module2", qconfig2)
            .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, qconfig3)

    '''
    global_qconfig: QConfigAny
    object_type_qconfigs: OrderedDict[Callable | str, QConfigAny]
    module_name_regex_qconfigs: OrderedDict[str, QConfigAny]
    module_name_qconfigs: OrderedDict[str, QConfigAny]
    module_name_object_type_order_qconfigs: OrderedDict[tuple[str, Callable, int], QConfigAny]
    def __init__(self) -> None: ...
    def set_global(self, global_qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the global (default) QConfig.
        """
    def set_object_type(self, object_type: Callable | str, qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the QConfig for a given module type, function, or method name.
        If the QConfig for an existing object type was already set, the new QConfig will override the old one.
        """
    def set_module_name_regex(self, module_name_regex: str, qconfig: QConfigAny) -> QConfigMapping:
        '''
        Set the QConfig for modules matching the given regex string.

        Regexes will be matched in the order in which they are registered through this method.
        Thus, the caller should register more specific patterns first, e.g.::

            qconfig_mapping = QConfigMapping()
                .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
                .set_module_name_regex("foo.*bar.*", qconfig2)
                .set_module_name_regex("foo.*", qconfig3)

        In this example, "foo.bar.conv0" would match qconfig1, "foo.bar.linear" would match qconfig2,
        and "foo.baz.relu" would match qconfig3.

        If the QConfig for an existing module name regex was already set, the new QConfig will override the
        old one while preserving the order in which the regexes were originally registered.
        '''
    def set_module_name(self, module_name: str, qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the QConfig for modules matching the given module name.
        If the QConfig for an existing module name was already set, the new QConfig will override the old one.
        """
    def set_module_name_object_type_order(self, module_name: str, object_type: Callable, index: int, qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the QConfig for modules matching a combination of the given module name, object type,
        and the index at which the module appears.

        If the QConfig for an existing (module name, object type, index)  was already set, the new QConfig
        will override the old one.
        """
    def __repr__(self) -> str: ...
    def to_dict(self) -> dict[str, Any]:
        '''
        Convert this ``QConfigMapping`` to a dictionary with the following keys:

            "" (for global QConfig)

            "object_type"

            "module_name_regex"

            "module_name"

            "module_name_object_type_order"

        The values of this dictionary are lists of tuples.
        '''
    @classmethod
    def from_dict(cls, qconfig_dict: dict[str, Any]) -> QConfigMapping:
        '''
        Create a ``QConfigMapping`` from a dictionary with the following keys (all optional):

            "" (for global QConfig)

            "object_type"

            "module_name_regex"

            "module_name"

            "module_name_object_type_order"

        The values of this dictionary are expected to be lists of tuples.
        '''
