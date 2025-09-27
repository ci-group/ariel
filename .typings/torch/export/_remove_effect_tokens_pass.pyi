from .exported_program import ExportedProgram as ExportedProgram
from .graph_signature import CustomObjArgument as CustomObjArgument, InputKind as InputKind, InputSpec as InputSpec, OutputKind as OutputKind, OutputSpec as OutputSpec, TokenArgument as TokenArgument
from torch._higher_order_ops.effects import _get_schema as _get_schema, with_effects as with_effects

def _remove_effect_tokens_from_graph_helper(ep, num_tokens, input_token_names, output_token_names) -> None: ...
def _remove_effect_tokens(ep: ExportedProgram) -> ExportedProgram:
    """
    Removes the existance of tokens from the exported program, including:
    - Removes the input and output tokens
    - Replaces with_effects(token, func, args) with just func(args)

    This function does an inplace modification on the given ExportedProgram.
    """
