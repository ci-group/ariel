from dataclasses import dataclass
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType
from typing import Any

__all__ = ['ConvertCustomConfig', 'FuseCustomConfig', 'PrepareCustomConfig', 'StandaloneModuleConfigEntry']

@dataclass
class StandaloneModuleConfigEntry:
    qconfig_mapping: QConfigMapping | None
    example_inputs: tuple[Any, ...]
    prepare_custom_config: PrepareCustomConfig | None
    backend_config: BackendConfig | None

class PrepareCustomConfig:
    '''
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.prepare_fx` and
    :func:`~torch.ao.quantization.quantize_fx.prepare_qat_fx`.

    Example usage::

        prepare_custom_config = PrepareCustomConfig()             .set_standalone_module_name("module1", qconfig_mapping, example_inputs,                 child_prepare_custom_config, backend_config)             .set_standalone_module_class(MyStandaloneModule, qconfig_mapping, example_inputs,                 child_prepare_custom_config, backend_config)             .set_float_to_observed_mapping(FloatCustomModule, ObservedCustomModule)             .set_non_traceable_module_names(["module2", "module3"])             .set_non_traceable_module_classes([NonTraceableModule1, NonTraceableModule2])             .set_input_quantized_indexes([0])             .set_output_quantized_indexes([0])             .set_preserved_attributes(["attr1", "attr2"])
    '''
    standalone_module_names: dict[str, StandaloneModuleConfigEntry]
    standalone_module_classes: dict[type, StandaloneModuleConfigEntry]
    float_to_observed_mapping: dict[QuantType, dict[type, type]]
    non_traceable_module_names: list[str]
    non_traceable_module_classes: list[type]
    input_quantized_indexes: list[int]
    output_quantized_indexes: list[int]
    preserved_attributes: list[str]
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def set_standalone_module_name(self, module_name: str, qconfig_mapping: QConfigMapping | None, example_inputs: tuple[Any, ...], prepare_custom_config: PrepareCustomConfig | None, backend_config: BackendConfig | None) -> PrepareCustomConfig:
        """
        Set the configuration for running a standalone module identified by ``module_name``.

        If ``qconfig_mapping`` is None, the parent ``qconfig_mapping`` will be used instead.
        If ``prepare_custom_config`` is None, an empty ``PrepareCustomConfig`` will be used.
        If ``backend_config`` is None, the parent ``backend_config`` will be used instead.
        """
    def set_standalone_module_class(self, module_class: type, qconfig_mapping: QConfigMapping | None, example_inputs: tuple[Any, ...], prepare_custom_config: PrepareCustomConfig | None, backend_config: BackendConfig | None) -> PrepareCustomConfig:
        """
        Set the configuration for running a standalone module identified by ``module_class``.

        If ``qconfig_mapping`` is None, the parent ``qconfig_mapping`` will be used instead.
        If ``prepare_custom_config`` is None, an empty ``PrepareCustomConfig`` will be used.
        If ``backend_config`` is None, the parent ``backend_config`` will be used instead.
        """
    def set_float_to_observed_mapping(self, float_class: type, observed_class: type, quant_type: QuantType = ...) -> PrepareCustomConfig:
        """
        Set the mapping from a custom float module class to a custom observed module class.

        The observed module class must have a ``from_float`` class method that converts the float module class
        to the observed module class. This is currently only supported for static quantization.
        """
    def set_non_traceable_module_names(self, module_names: list[str]) -> PrepareCustomConfig:
        """
        Set the modules that are not symbolically traceable, identified by name.
        """
    def set_non_traceable_module_classes(self, module_classes: list[type]) -> PrepareCustomConfig:
        """
        Set the modules that are not symbolically traceable, identified by class.
        """
    def set_input_quantized_indexes(self, indexes: list[int]) -> PrepareCustomConfig:
        """
        Set the indexes of the inputs of the graph that should be quantized.
        Inputs are otherwise assumed to be in fp32 by default instead.
        """
    def set_output_quantized_indexes(self, indexes: list[int]) -> PrepareCustomConfig:
        """
        Set the indexes of the outputs of the graph that should be quantized.
        Outputs are otherwise assumed to be in fp32 by default instead.
        """
    def set_preserved_attributes(self, attributes: list[str]) -> PrepareCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
    @classmethod
    def from_dict(cls, prepare_custom_config_dict: dict[str, Any]) -> PrepareCustomConfig:
        '''
        Create a ``PrepareCustomConfig`` from a dictionary with the following items:

            "standalone_module_name": a list of (module_name, qconfig_mapping, example_inputs,
            child_prepare_custom_config, backend_config) tuples

            "standalone_module_class" a list of (module_class, qconfig_mapping, example_inputs,
            child_prepare_custom_config, backend_config) tuples

            "float_to_observed_custom_module_class": a nested dictionary mapping from quantization
            mode to an inner mapping from float module classes to observed module classes, e.g.
            {"static": {FloatCustomModule: ObservedCustomModule}}

            "non_traceable_module_name": a list of modules names that are not symbolically traceable
            "non_traceable_module_class": a list of module classes that are not symbolically traceable
            "input_quantized_idxs": a list of indexes of graph inputs that should be quantized
            "output_quantized_idxs": a list of indexes of graph outputs that should be quantized
            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        '''
    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``PrepareCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict`.
        """

class ConvertCustomConfig:
    '''
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.convert_fx`.

    Example usage::

        convert_custom_config = ConvertCustomConfig()             .set_observed_to_quantized_mapping(ObservedCustomModule, QuantizedCustomModule)             .set_preserved_attributes(["attr1", "attr2"])
    '''
    observed_to_quantized_mapping: dict[QuantType, dict[type, type]]
    preserved_attributes: list[str]
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def set_observed_to_quantized_mapping(self, observed_class: type, quantized_class: type, quant_type: QuantType = ...) -> ConvertCustomConfig:
        """
        Set the mapping from a custom observed module class to a custom quantized module class.

        The quantized module class must have a ``from_observed`` class method that converts the observed module class
        to the quantized module class.
        """
    def set_preserved_attributes(self, attributes: list[str]) -> ConvertCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
    @classmethod
    def from_dict(cls, convert_custom_config_dict: dict[str, Any]) -> ConvertCustomConfig:
        '''
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "observed_to_quantized_custom_module_class": a nested dictionary mapping from quantization
            mode to an inner mapping from observed module classes to quantized module classes, e.g.::
            {
            "static": {FloatCustomModule: ObservedCustomModule},
            "dynamic": {FloatCustomModule: ObservedCustomModule},
            "weight_only": {FloatCustomModule: ObservedCustomModule}
            }
            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        '''
    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``ConvertCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """

class FuseCustomConfig:
    '''
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.fuse_fx`.

    Example usage::

        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(
            ["attr1", "attr2"]
        )
    '''
    preserved_attributes: list[str]
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def set_preserved_attributes(self, attributes: list[str]) -> FuseCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
    @classmethod
    def from_dict(cls, fuse_custom_config_dict: dict[str, Any]) -> FuseCustomConfig:
        '''
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        '''
    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``FuseCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """
