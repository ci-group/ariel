from _typeshed import Incomplete
from torch.distributed._shard.sharded_tensor import _sharded_op_impl as _sharded_op_impl

def validate_param(param, param_name) -> None: ...
def uniform_(types, args=(), kwargs=None, pg=None):
    """
    Fills the Tensor in tensor.local_shards with values drawn from the uniform
    distribution :math:`\\mathcal{U}(a, b)`.
    Args:
        tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
def normal_(types, args=(), kwargs=None, pg=None):
    """
    Fills the Tensors in tensor.local_shards with values drawn from the normal
    distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`.
    Args:
        tensor: tensor sharded across devices
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    """
def kaiming_uniform_(types, args=(), kwargs=None, pg=None):
    """
    Fills the Tensors in tensor.local_shards with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\\mathcal{U}(-\\text{bound}, \\text{bound})` where
    .. math::
        \\text{bound} = \\text{gain} \\times \\sqrt{\\frac{3}{\\text{fan\\_mode}}}
    Also known as He initialization.
    Args:
        tensor: tensor sharded across devices
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    """
def constant_(types, args=(), kwargs=None, pg=None):
    """
    Fills the input ShardedTensor with the value \\text{val}val.
    Args:
        tensor: tensor sharded across devices
        val: the value to fill the tensor with
    """

tensor_like_creation_op_map: Incomplete

def register_tensor_creation_op(op): ...
