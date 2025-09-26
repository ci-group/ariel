import torch
import torch.distributed as dist
from _typeshed import Incomplete

__all__ = ['PowerSGDState', 'powerSGD_hook', 'batched_powerSGD_hook']

class PowerSGDState:
    """
    Store both the algorithm's hyperparameters and internal state for all gradients during training.

    Particularly, ``matrix_approximation_rank`` and ``start_powerSGD_iter`` are the main hyperparameters that should be tuned by the user.
    For performance, we suggest to keep binary hyperparameters ``use_error_feedback`` and ``warm_start`` on.

    1. ``matrix_approximation_rank`` controls the size of compressed low-rank tensors, which determines the compression rate. The lower the rank, the stronger the compression.

        1.1. If ``matrix_approximation_rank`` is too low, the full model quality will need more training steps to reach or will never reach and yield loss in accuracy.

        1.2. The increase of ``matrix_approximation_rank`` can substantially increase the computation costs of the compression, and the accuracy may not be further improved beyond a certain ``matrix_approximation_rank`` threshold.

    To tune ``matrix_approximation_rank``, we suggest to start from 1 and increase by factors of 2 (like an exponential grid search, 1, 2, 4, ...), until a satisfactory accuracy is reached. Typically only a small value 1-4 is used. For some NLP tasks (as shown in Appendix D of the original paper), this value has been increased to 32.

    2. ``start_powerSGD_iter`` defers PowerSGD compression until step ``start_powerSGD_iter``, and vanilla allreduce runs prior to step ``start_powerSGD_iter``. This hybrid scheme of **vanilla allreduce + PowerSGD** can effectively improve the accuracy, even a relatively small ``matrix_approximation_rank`` is used. This is because that, the beginning of training phase is usually very sensitive to inaccurate gradients, and compressing gradients too early may make the training quickly take a suboptimal trajectory, which can result in an irrecoverable impact on the accuracy.

    To tune ``start_powerSGD_iter``, we suggest to start with 10% of total training steps, and increase it until a satisfactory accuracy is reached. If there is a warm-up stage in the training, ``start_powerSGD_iter`` typically should be no less than the number of warm-up steps.

    3. ``min_compression_rate`` is the minimum compression rate required when a layer is compressed. Due to the computation overheads incurred by the compression, a tensor is worth compressing only if there can be sufficient saving in bandwidth, where ``(num_rows + num_cols) * matrix_approximation_rank * min_compression_rate < num_rows * num_cols``. If the specified compression rate threshold cannot be satisfied, the tensor will be directly allreduced without compression.

    Compression statistics are logged every ``compression_stats_logging_frequency`` iterations once PowerSGD compression starts.

    4. ``orthogonalization_epsilon`` can be a very small value (e.g., 1e-8) added to every normalized matrix column in orthogonalization step, to prevent div-by-zero error if any column has all 0s. If this can already be prevented (e.g., by batch normalization), an epsilon of 0 is recommended for accuracy.

    5. ``batch_tensors_with_same_shape`` controls whether to compress and decompress tensors with same shape in a batched operation to achieve higher parallelism. Note that you should also increase the bucket size (i.e., ``bucket_cap_mb`` arg in DDP constructor) to make more same-shaped tensors appear in the same bucket, however this may reduce the overlap between computation and communication, and increase the memory footprint due to stacking the tensors of the same shape. Set to ``True`` if the compression / decompression computation is a bottleneck.

    .. warning ::
        If error feedback or warm-up is enabled, the minimum value of ``start_powerSGD_iter`` allowed in DDP is 2.
        This is because there is another internal optimization that rebuilds buckets at iteration 1 in DDP,
        and this can conflict with any tensor memorized before the rebuild process.
    """
    __slots__: Incomplete
    process_group: Incomplete
    matrix_approximation_rank: Incomplete
    start_powerSGD_iter: Incomplete
    min_compression_rate: Incomplete
    use_error_feedback: Incomplete
    warm_start: Incomplete
    orthogonalization_epsilon: Incomplete
    rng: Incomplete
    error_dict: dict[int, torch.Tensor]
    p_memory_dict: dict[int, torch.Tensor]
    q_memory_dict: dict[int, torch.Tensor]
    iter: int
    total_numel_before_compression: int
    total_numel_after_compression: int
    compression_stats_logging_frequency: Incomplete
    next_stats_report: int
    batch_tensors_with_same_shape: Incomplete
    def __init__(self, process_group, matrix_approximation_rank: int = 1, start_powerSGD_iter: int = 1000, min_compression_rate: int = 2, use_error_feedback: bool = True, warm_start: bool = True, orthogonalization_epsilon: int = 0, random_seed: int = 0, compression_stats_logging_frequency: int = 10000, batch_tensors_with_same_shape: bool = False) -> None: ...
    def __getstate__(self):
        """
        Return a ``Dict[str, Any]`` which will be pickled and saved.

        ``process_group`` is not serializable and excluded from
        a returned state.
        """
    def __setstate__(self, state) -> None:
        """
        Take a provided ``state`` and set to this ``PowerSGDState`` instance.

        ``process_group`` is set to default.
        """
    def maybe_increase_iter(self, bucket) -> None:
        """Track iterations and trigger log message at start of local SGD."""
    def compression_stats(self):
        """
        Return latest compression statistics as tuple.

        Returns tuple of form (compress_rate, numel_before_compression, numel_after_compression) where:

        compress_rate is the effective compression rate i.e. (number of elements before compression) / (number of elements after compression);

        numel_before_compression is the total number of elements before compression was applied; and,

        numel_after_compression is the total number of elements after compression was applied.
        """

def powerSGD_hook(state: PowerSGDState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Implement PowerSGD algorithm.

    This DDP communication hook implements PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:

    1. Views the input flattened 1D gradient tensor as a list of per-parameter tensors, and divides all the tensors into two groups:

        1.1 The tensors that should be compressed before allreduce, because the compression can give enough saving in bandwidth.

        1.2 Rest of the tensors will be directly allreduced without compression, including all the vector tensors (for biases).

    2. Handles uncompressed tensors:

        2.1. Allocate contiguous memory for those uncompressed tensors, and allreduces all the uncompressed tensors as a batch, without compression;

        2.2. Copies the individual uncompressed tensors from the contiguous memory back to the input tensor.

    3. Handles the tensors that should be compressed by PowerSGD compression:

        3.1. For each tensor M, creates two low-rank tensors P and Q for decomposing M,
        such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;

        3.2. Computes each P in Ps, which is equal to MQ;

        3.3. Allreduces Ps as a batch;

        3.4. Orthogonalizes each P in Ps;

        3.5. Computes each Q in Qs, which is approximately equal to M^TP;

        3.6. Allreduces Qs as a batch;

        3.7. Computes each M among all the compressed tensors, which is approximately equal to PQ^T.

    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.
    This not only gives the user more control over the tradeoff between speedup and accuracy,
    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, mainly need to tune ``matrix_approximation_rank``, ``start_powerSGD_iter``
            and ``min_compression_rate``.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1,
                                  start_powerSGD_iter=10, min_compression_rate=0.5)
        >>> ddp_model.register_comm_hook(state, powerSGD_hook)
    """
def batched_powerSGD_hook(state: PowerSGDState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Implement simplified PowerSGD algorithm.

    This DDP communication hook implements a simplified PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.
    This variant does not compress the gradients layer by layer,
    but instead compresses the flattened input tensor that batches all the gradients.
    Therefore, it is **faster** than :meth:`powerSGD_hook`,
    but usually results in a **much lower accuracy**, unless ``matrix_approximation_rank`` is 1.

    .. warning ::
        Increasing ``matrix_approximation_rank`` here may not necessarily increase the accuracy,
        because batching per-parameter tensors without column/row alignment can destroy low-rank structure.
        Therefore, the user should always consider :meth:`powerSGD_hook` first,
        and only consider this variant when a satisfactory accuracy can be achieved when ``matrix_approximation_rank`` is 1.

    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:

    1. Views the input flattened 1D gradient tensor as a square-shaped tensor M with 0 paddings;

    2. Creates two low-rank tensors P and Q for decomposing M, such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;

    3. Computes P, which is equal to MQ;

    4. Allreduces P;

    5. Orthogonalizes P;

    6. Computes Q, which is approximately equal to M^TP;

    7. Allreduces Q;

    8. Computes M, which is approximately equal to PQ^T.

    9. Truncates the input tensor to the original length.

    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.
    This not only gives the user more control over the tradeoff between speedup and accuracy,
    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, mainly need to tune ``matrix_approximation_rank`` and ``start_powerSGD_iter``.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1)
        >>> ddp_model.register_comm_hook(state, batched_powerSGD_hook)
    """
