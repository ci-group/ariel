import dataclasses
from collections.abc import Collection, Mapping
from enum import Enum
from torch._library.fake_class_registry import FakeScriptObject

__all__ = ['ConstantArgument', 'CustomObjArgument', 'ExportBackwardSignature', 'ExportGraphSignature', 'InputKind', 'InputSpec', 'OutputKind', 'OutputSpec', 'SymIntArgument', 'SymFloatArgument', 'SymBoolArgument', 'TensorArgument']

@dataclasses.dataclass
class TensorArgument:
    name: str

@dataclasses.dataclass
class TokenArgument:
    name: str

@dataclasses.dataclass
class SymIntArgument:
    name: str

@dataclasses.dataclass
class SymFloatArgument:
    name: str

@dataclasses.dataclass
class SymBoolArgument:
    name: str

@dataclasses.dataclass
class CustomObjArgument:
    name: str
    class_fqn: str
    fake_val: FakeScriptObject | None = ...

@dataclasses.dataclass
class ConstantArgument:
    name: str
    value: int | float | bool | str | None
ArgumentSpec = TensorArgument | SymIntArgument | SymFloatArgument | SymBoolArgument | ConstantArgument | CustomObjArgument | TokenArgument

class InputKind(Enum):
    USER_INPUT = ...
    PARAMETER = ...
    BUFFER = ...
    CONSTANT_TENSOR = ...
    CUSTOM_OBJ = ...
    TOKEN = ...

@dataclasses.dataclass
class InputSpec:
    kind: InputKind
    arg: ArgumentSpec
    target: str | None
    persistent: bool | None = ...
    def __post_init__(self) -> None: ...
    def __str__(self) -> str: ...

class OutputKind(Enum):
    USER_OUTPUT = ...
    LOSS_OUTPUT = ...
    BUFFER_MUTATION = ...
    GRADIENT_TO_PARAMETER = ...
    GRADIENT_TO_USER_INPUT = ...
    USER_INPUT_MUTATION = ...
    TOKEN = ...

@dataclasses.dataclass
class OutputSpec:
    kind: OutputKind
    arg: ArgumentSpec
    target: str | None
    def __post_init__(self) -> None: ...
    def __str__(self) -> str: ...

@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: dict[str, str]
    gradients_to_user_inputs: dict[str, str]
    loss_output: str

@dataclasses.dataclass
class ExportGraphSignature:
    '''
    :class:`ExportGraphSignature` models the input/output signature of Export Graph,
    which is a fx.Graph with stronger invariants gurantees.

    Export Graph is functional and does not access "states" like parameters
    or buffers within the graph via ``getattr`` nodes. Instead, :func:`export`
    gurantees that parameters, buffers, and constant tensors are lifted out of
    the graph as inputs.  Similarly, any mutations to buffers are not included
    in the graph either, instead the updated values of mutated buffers are
    modeled as additional outputs of Export Graph.

    The ordering of all inputs and outputs are::

        Inputs = [*parameters_buffers_constant_tensors, *flattened_user_inputs]
        Outputs = [*mutated_inputs, *flattened_user_outputs]

    e.g. If following module is exported::

        class CustomModule(nn.Module):
            def __init__(self) -> None:
                super(CustomModule, self).__init__()

                # Define a parameter
                self.my_parameter = nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)  # In-place addition

                return output


        mod = CustomModule()
        ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))

    Resulting Graph is non-functional::

        graph():
            %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
            %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
            %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
            %x1 : [num_users=1] = placeholder[target=x1]
            %x2 : [num_users=1] = placeholder[target=x2]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
            %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
            %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
            %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
            %add_ : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
            return (add_1,)

    Resulting ExportGraphSignature of the non-functional Graph would be::

        # inputs
        p_my_parameter: PARAMETER target=\'my_parameter\'
        b_my_buffer1: BUFFER target=\'my_buffer1\' persistent=True
        b_my_buffer2: BUFFER target=\'my_buffer2\' persistent=True
        x1: USER_INPUT
        x2: USER_INPUT

        # outputs
        add_1: USER_OUTPUT

    To get a functional Graph, you can use :func:`run_decompositions`::

        mod = CustomModule()
        ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))
        ep = ep.run_decompositions()

    Resulting Graph is functional::

        graph():
            %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
            %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
            %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
            %x1 : [num_users=1] = placeholder[target=x1]
            %x2 : [num_users=1] = placeholder[target=x2]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
            %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
            %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
            %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
            %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
            return (add_2, add_1)

    Resulting ExportGraphSignature of the functional Graph would be::

        # inputs
        p_my_parameter: PARAMETER target=\'my_parameter\'
        b_my_buffer1: BUFFER target=\'my_buffer1\' persistent=True
        b_my_buffer2: BUFFER target=\'my_buffer2\' persistent=True
        x1: USER_INPUT
        x2: USER_INPUT

        # outputs
        add_2: BUFFER_MUTATION target=\'my_buffer2\'
        add_1: USER_OUTPUT

    '''
    input_specs: list[InputSpec]
    output_specs: list[OutputSpec]
    @property
    def parameters(self) -> Collection[str]: ...
    @property
    def buffers(self) -> Collection[str]: ...
    @property
    def non_persistent_buffers(self) -> Collection[str]: ...
    @property
    def lifted_tensor_constants(self) -> Collection[str]: ...
    @property
    def lifted_custom_objs(self) -> Collection[str]: ...
    @property
    def user_inputs(self) -> Collection[int | float | bool | None | str]: ...
    @property
    def user_outputs(self) -> Collection[int | float | bool | None | str]: ...
    @property
    def inputs_to_parameters(self) -> Mapping[str, str]: ...
    @property
    def inputs_to_buffers(self) -> Mapping[str, str]: ...
    @property
    def buffers_to_mutate(self) -> Mapping[str, str]: ...
    @property
    def user_inputs_to_mutate(self) -> Mapping[str, str]: ...
    @property
    def inputs_to_lifted_tensor_constants(self) -> Mapping[str, str]: ...
    @property
    def inputs_to_lifted_custom_objs(self) -> Mapping[str, str]: ...
    @property
    def backward_signature(self) -> ExportBackwardSignature | None: ...
    @property
    def assertion_dep_token(self) -> Mapping[int, str] | None: ...
    @property
    def input_tokens(self) -> Collection[str]: ...
    @property
    def output_tokens(self) -> Collection[str]: ...
    def __post_init__(self) -> None: ...
    def replace_all_uses(self, old: str, new: str):
        """
        Replace all uses of the old name with new name in the signature.
        """
    def get_replace_hook(self, replace_inputs: bool = False): ...
    def __str__(self) -> str: ...
