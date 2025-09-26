from .functional_utils import MetadataKey as MetadataKey, are_all_mutations_hidden_from_autograd as are_all_mutations_hidden_from_autograd, are_all_mutations_under_no_grad_or_inference_mode as are_all_mutations_under_no_grad_or_inference_mode, from_fun as from_fun, has_data_mutation as has_data_mutation, has_metadata_mutation as has_metadata_mutation, to_fun as to_fun, was_inductor_storage_resized as was_inductor_storage_resized
from .schemas import FunctionalTensorMetadataEq as FunctionalTensorMetadataEq, InputAliasInfo as InputAliasInfo, MemoryFormatMeta as MemoryFormatMeta, MutationType as MutationType, OutputAliasInfo as OutputAliasInfo, OutputType as OutputType, ViewAndMutationMeta as ViewAndMutationMeta
from .subclass_utils import create_subclass_meta as create_subclass_meta
from .utils import KNOWN_TYPES as KNOWN_TYPES, _get_autocast_states as _get_autocast_states, strict_zip as strict_zip
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch._guards import detect_fake_mode as detect_fake_mode
from torch._logging import getArtifactLogger as getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor as FunctionalTensor, FunctionalTensorMode as FunctionalTensorMode
from torch._subclasses.meta_utils import safe_is_leaf as safe_is_leaf
from torch.fx.experimental.symbolic_shapes import is_concrete_int as is_concrete_int
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass, transform_subclass as transform_subclass
from typing import Callable

zip = strict_zip
log: Incomplete
static_input_logger: Incomplete

def coerce_tangent_and_suggest_memory_format(x: Tensor): ...
def run_functionalized_fw_and_collect_metadata(f, *, keep_input_mutations: bool, is_train: bool = False, static_input_indices: list[int] | None = None, pre_dispatch: bool = False, is_export: bool = False) -> Callable[..., ViewAndMutationMeta]: ...
