from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any, Callable

__all__ = ['Fuzzer', 'FuzzedParameter', 'ParameterAlias', 'FuzzedTensor']

class FuzzedParameter:
    """Specification for a parameter to be generated during fuzzing."""
    _name: Incomplete
    _minval: Incomplete
    _maxval: Incomplete
    _distribution: Incomplete
    strict: Incomplete
    def __init__(self, name: str, minval: int | float | None = None, maxval: int | float | None = None, distribution: str | dict[Any, float] | None = None, strict: bool = False) -> None:
        '''
        Args:
            name:
                A string name with which to identify the parameter.
                FuzzedTensors can reference this string in their
                specifications.
            minval:
                The lower bound for the generated value. See the description
                of `distribution` for type behavior.
            maxval:
                The upper bound for the generated value. Type behavior is
                identical to `minval`.
            distribution:
                Specifies the distribution from which this parameter should
                be drawn. There are three possibilities:
                    - "loguniform"
                        Samples between `minval` and `maxval` (inclusive) such
                        that the probabilities are uniform in log space. As a
                        concrete example, if minval=1 and maxval=100, a sample
                        is as likely to fall in [1, 10) as it is [10, 100].
                    - "uniform"
                        Samples are chosen with uniform probability between
                        `minval` and `maxval` (inclusive). If either `minval`
                        or `maxval` is a float then the distribution is the
                        continuous uniform distribution; otherwise samples
                        are constrained to the integers.
                    - dict:
                        If a dict is passed, the keys are taken to be choices
                        for the variables and the values are interpreted as
                        probabilities. (And must sum to one.)
                If a dict is passed, `minval` and `maxval` must not be set.
                Otherwise, they must be set.
            strict:
                If a parameter is strict, it will not be included in the
                iterative resampling process which Fuzzer uses to find a
                valid parameter configuration. This allows an author to
                prevent skew from resampling for a given parameter (for
                instance, a low size limit could inadvertently bias towards
                Tensors with fewer dimensions) at the cost of more iterations
                when generating parameters.
        '''
    @property
    def name(self): ...
    def sample(self, state): ...
    def _check_distribution(self, distribution): ...
    def _loguniform(self, state): ...
    def _uniform(self, state): ...
    def _custom_distribution(self, state): ...

class ParameterAlias:
    '''Indicates that a parameter should alias the value of another parameter.

    When used in conjunction with a custom distribution, this allows fuzzed
    tensors to represent a broader range of behaviors. For example, the
    following sometimes produces Tensors which broadcast:

    Fuzzer(
        parameters=[
            FuzzedParameter("x_len", 4, 1024, distribution="uniform"),

            # `y` will either be size one, or match the size of `x`.
            FuzzedParameter("y_len", distribution={
                0.5: 1,
                0.5: ParameterAlias("x_len")
            }),
        ],
        tensors=[
            FuzzedTensor("x", size=("x_len",)),
            FuzzedTensor("y", size=("y_len",)),
        ],
    )

    Chains of alias\' are allowed, but may not contain cycles.
    '''
    alias_to: Incomplete
    def __init__(self, alias_to) -> None: ...
    def __repr__(self) -> str: ...

class FuzzedTensor:
    _name: Incomplete
    _size: Incomplete
    _steps: Incomplete
    _probability_contiguous: Incomplete
    _min_elements: Incomplete
    _max_elements: Incomplete
    _max_allocation_bytes: Incomplete
    _dim_parameter: Incomplete
    _dtype: Incomplete
    _cuda: Incomplete
    _tensor_constructor: Incomplete
    def __init__(self, name: str, size: tuple[str | int, ...], steps: tuple[str | int, ...] | None = None, probability_contiguous: float = 0.5, min_elements: int | None = None, max_elements: int | None = None, max_allocation_bytes: int | None = None, dim_parameter: str | None = None, roll_parameter: str | None = None, dtype=..., cuda: bool = False, tensor_constructor: Callable | None = None) -> None:
        """
        Args:
            name:
                A string identifier for the generated Tensor.
            size:
                A tuple of integers or strings specifying the size of the generated
                Tensor. String values will replaced with a concrete int during the
                generation process, while ints are simply passed as literals.
            steps:
                An optional tuple with the same length as `size`. This indicates
                that a larger Tensor should be allocated, and then sliced to
                produce the generated Tensor. For instance, if size is (4, 8)
                and steps is (1, 4), then a tensor `t` of size (4, 32) will be
                created and then `t[:, ::4]` will be used. (Allowing one to test
                Tensors with strided memory.)
            probability_contiguous:
                A number between zero and one representing the chance that the
                generated Tensor has a contiguous memory layout. This is achieved by
                randomly permuting the shape of a Tensor, calling `.contiguous()`,
                and then permuting back. This is applied before `steps`, which can
                also cause a Tensor to be non-contiguous.
            min_elements:
                The minimum number of parameters that this Tensor must have for a
                set of parameters to be valid. (Otherwise they are resampled.)
            max_elements:
                Like `min_elements`, but setting an upper bound.
            max_allocation_bytes:
                Like `max_elements`, but for the size of Tensor that must be
                allocated prior to slicing for `steps` (if applicable). For
                example, a FloatTensor with size (1024, 1024) and steps (4, 4)
                would have 1M elements, but would require a 64 MB allocation.
            dim_parameter:
                The length of `size` and `steps` will be truncated to this value.
                This allows Tensors of varying dimensions to be generated by the
                Fuzzer.
            dtype:
                The PyTorch dtype of the generated Tensor.
            cuda:
                Whether to place the Tensor on a GPU.
            tensor_constructor:
                Callable which will be used instead of the default Tensor
                construction method. This allows the author to enforce properties
                of the Tensor (e.g. it can only have certain values). The dtype and
                concrete shape of the Tensor to be created will be passed, and
                concrete values of all parameters will be passed as kwargs. Note
                that transformations to the result (permuting, slicing) will be
                performed by the Fuzzer; the tensor_constructor is only responsible
                for creating an appropriately sized Tensor.
        """
    @property
    def name(self): ...
    @staticmethod
    def default_tensor_constructor(size, dtype, **kwargs): ...
    def _make_tensor(self, params, state): ...
    def _get_size_and_steps(self, params): ...
    def satisfies_constraints(self, params): ...

class Fuzzer:
    _seed: Incomplete
    _parameters: Incomplete
    _tensors: Incomplete
    _constraints: Incomplete
    _rejections: int
    _total_generated: int
    def __init__(self, parameters: list[FuzzedParameter | list[FuzzedParameter]], tensors: list[FuzzedTensor | list[FuzzedTensor]], constraints: list[Callable] | None = None, seed: int | None = None) -> None:
        """
        Args:
            parameters:
                List of FuzzedParameters which provide specifications
                for generated parameters. Iterable elements will be
                unpacked, though arbitrary nested structures will not.
            tensors:
                List of FuzzedTensors which define the Tensors which
                will be created each step based on the parameters for
                that step. Iterable elements will be unpacked, though
                arbitrary nested structures will not.
            constraints:
                List of callables. They will be called with params
                as kwargs, and if any of them return False the current
                set of parameters will be rejected.
            seed:
                Seed for the RandomState used by the Fuzzer. This will
                also be used to set the PyTorch random seed so that random
                ops will create reproducible Tensors.
        """
    @staticmethod
    def _unpack(values, cls): ...
    def take(self, n) -> Generator[Incomplete]: ...
    @property
    def rejection_rate(self): ...
    def _generate(self, state): ...
    @staticmethod
    def _resolve_aliases(params): ...
