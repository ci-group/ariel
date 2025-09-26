from ._common import _all_gather_base_input as _all_gather_base_input, _handle_col_wise_sharding_base as _handle_col_wise_sharding_base, _handle_max_norm_col_wise as _handle_max_norm_col_wise, _handle_row_wise_mask as _handle_row_wise_mask
from torch.distributed._shard.sharded_tensor import ShardedTensor as ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec as ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op as custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather as all_gather, reduce_scatter as reduce_scatter

def sharded_embedding(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding``.
    This method computes a sharded embedding lookup and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Supports all specs except for scale_grad_by_freq, sparse, etc.

    Based on the dimension that the weight is sharded on, there are two
    algorithms:

    ROWWISE SHARDING
    ================
    For row-wise sharding the weight is sharded on dimension 0.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (10 x 17) and W is sharded across
    4 GPUs creating 3 shard of (3 x 17) and 1 shard of (1 x 17).
    The algorithm is as follows:

    1. First the input is all gathered to all ranks, since this is SPMD and
       input is actually sharded across all ranks. The inputs then become a
       4 (4 x 6) tensor on each rank. For example if the given input is
       tensor([[6, 5, 2, 9, 6, 3],
               [3, 1, 2, 4, 7, 6],
               [4, 0, 4, 9, 8, 9],
               [8, 6, 6, 4, 6, 1]])
       on rank 0.
       Then on every rank, we will have this tensor.
       If input itself is already replicated, no all-gather will be done.
    2. Next, we mask the ID which are not stored on that rank.
       For example on rank 0, we store ID [0, 1, 2]. We only keep the ID
       inside the set of numbers. The rest of them will be masked to an extra row.
       The masked matrix will be used for embedding look up and is like:
       tensor([[4, 4, 2, 4, 4, 4],
               [4, 1, 2, 4, 4, 4],
               [4, 0, 4, 4, 4, 4],
               [4, 4, 4, 4, 4, 1]])
       The reason of having an extra row (aka, number 4 in the example) is
       because when max_norm is specified only weight which has looked will
       be re-normed so mask IDs whose embeddings are not stored in current
       rank will to an extra row will ensure max_norm still works as expected.
    3. If max_norm is specified, the extra row guarantees that the mask ID will
       not affect the behavior of weigh re-norm.

    COLWISE SHARDING
    ================
    For col-wise sharding the weight is sharded on dimension 1.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 3 shards of (16 x 5) and 1 shard of (16 x 2).
    The algorithm is as follows:

    1. First the input is broadcasted to all ranks, since this is SPMD we
       actually do an all_gather for all the inputs resulting in 4 (4 x 6)
       inputs on each rank.
    2. Next we perform local embedding lookup operation by apply each
       input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).
       This results in 4 (5 x 6 x 4) ((2 x 6 x 4) for the last) matrices
       on each rank. We transpose dim 0 and dim 2.
    3. Next, we concat these 4 matrices and perform an all2all to share the
       appropriate (5 x 6 x 4) or (2 x 6 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 6 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 6 x 4) matrix and finally we transpose the
       dim 0 and dim 2 again.
    6. If max_norm is specified, we manually sum up the norm and renorm. Because
       the renorm must be in place, we need to override the local_shard to mimic
       this behavior.
    """
def _validate_embedding_param(args, kwargs) -> None:
    """
    Validate input params of sharded embedding op.

    Args:
        input: list of ID used for lookup.
        weight: sharded weight tensor.
        kwargs: same as normal Embedding.

    Return: None.
    """
def _handle_col_wise_sharding(input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, pg):
    '''
    Entry-point function to handle the logic of col-wise sharding of weight
    for embedding. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise shared local weight used for lookup.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
        pg: process group.

    Returns: final result of lookup.
    '''
def _handle_row_wise_sharding(input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, rank, pg):
    '''
    Entry-point function to handle the logic of row-wise sharding of weight
    for embedding. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: row-wise shared local weight used for lookup.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
        rank: # of cuda process.
        pg: process group.

    Returns: final result of lookup.
    '''
