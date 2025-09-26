from torch.distributed._shard.sharded_tensor import ShardedTensor as ShardedTensor
from torch.distributed._shard.sharded_tensor._ops._common import _sharded_op_common as _sharded_op_common
from torch.distributed._shard.sharding_spec import ChunkShardingSpec as ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import get_chunk_sharding_params as get_chunk_sharding_params, get_chunked_dim_size as get_chunked_dim_size, get_split_size as get_split_size
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op as custom_sharding_spec_op
from torch.distributed.nn.functional import _all_gather_base as _all_gather_base, all_reduce as all_reduce, all_to_all_single as all_to_all_single

def _chunk_sharding_spec_check(spec, op) -> None:
    """
    For the given op implementation check if the sharding spec is ChunkShardingSpec.
    """
def _register_sharded_op_on_local_tensor(op, early_stop_func=None, extra_check=None, customized_func=None):
    """
    Handles ``__torch_function__`` dispatch for ops which are performed on
    the single local tensor of the sharded tensor such as op like
    ``torch.nn.functional.softmax`` or ``torch.Tensor.view``.

    For more complicated ops, a customized func can be used to generate
    the new local tensor, sharding spec and sharded tensor size.

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.
        customized_func (Callable, optional): the func for customized logic
            to generate the new local tensor, sharding spec and sharded tensor size.
            Default: if ``None``, we simply lower to the real op call with
                the single local tensor of the st.

    Return:
        func (Callable): registered implementation for sharded op for
        ``__torch_function__`` dispatch.
    """
def _handle_col_wise_sharding_base(op_func, col_dim, input, world_size, weight, local_shard, pg, gathered_inputs, mode=None, gathered_per_sample_weights=None, gathered_offsets=None, padding_idx=None):
    '''
    For col-wise sharding of weight, lots of logic are common.
    So we extract the common logic and put in this function:
    Step 1. To get input from each rank and
    Step 2. To perform the op on the concatenated tensor.
    Step 3. To distribute results to each rank with col rearrangement.
    Step 4. To concatenate all results from all ranks.

    Args:
        op_func: operator which is applied to the input tensor.
        col_dim: dim of result tensor after the operation.
        input: tensor to be applied op on.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise sharded weight tensor.
        pg: process group.
        gathered_inputs: list of inputs from all ranks. If specified, we
            don\'t need to communicate with each rank any more.
        mode: aggregation mode of EmbeddingBag.
        gathered_per_sample_weights: per_sample_weights across all ranks.
        gathered_offsets: offsets across all ranks.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.

    Return: final result of input being applied with the op.
    '''
def _result_distribute_with_col_rearrange(results, input, world_size, weight, pg):
    """
    For col-wise sharding of weight, we need to distribute
    results to each rank. We do them in this function.
    Note that, if the index in the Sharding Spec is not equal to
    the rank number, we need to do the rearrangement based on the
    order given by the Sharding Spec (placement).

    Args:
        results: results from ops applied to inputs from all ranks.
            We need to distribute them back to their original ranks.
        input: tensor to be applied op to.
        world_size: number of ranks.
        weight: sharded weight tensor.
        pg: process group.

    Return: column rearranged result.
    """
def _handle_max_norm_col_wise(max_norm, norm_type, local_shard, input, world_size, gathered_inputs, pg):
    """
    For col-wise sharding of weight, we need to aggregate the
    norm across all ranks before we can perform the proper re-norm.
    Note that, the max_norm logic is only applied to the embedding
    indices that are looked up and not the whole shard.

    Args:
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        local_shard: col-wise shared local weight used for lookup.
        input: tensor to be applied op to.
        world_size: number of ranks.
        gathered_inputs: list of inputs from all ranks.
        pg: process group.

    Return:
        local_shard_norm_renormed: local_shard re-normed to max_norm if the norm is larger
            than it.

    """
def _all_gather_base_input(input, pg):
    """
    Use _all_gather_base to get a concatenated input from each rank.

    Args:
        input: tensor to be applied op on.
        pg: process group.

    Returns:
        gathered_inputs: input gathered from each rank and concat by dim 0.
    """
def _handle_row_wise_mask(gather_inp, padding_idx, weight, world_size, rank):
    '''
    Mask the input for embedding look-up for IDs which are not stored
    on the current rank. This function also adjust the ``padding_idx``
    so that it is only used on the rank where the corresponding row is
    stored.

    Note that, with ``max_norm`` flag on, only weights of rows being
    looked up will be re-normed. So we need an extra row for masked ID
    so that it does not affect the final result and ``max_norm``.

    Args:
        gather_inp: tensor to be applied op on gathered from all ranks.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        weight: weight tensor of Embedding look-up table.
        world_size: number of ranks.
        rank: # of cuda process.

    Returns:
        lookup_input: Tensor of masked input.
        padding_idx: adjusted padding_idx.
        padding_row: The extra row we used during lookup so that
            looking up does not affect ``max_norm``.
    '''
