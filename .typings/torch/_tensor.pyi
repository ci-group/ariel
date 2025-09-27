import enum
import torch
import torch._C as _C
from _typeshed import Incomplete
from torch._namedtensor_internals import check_serializing_named_tensor as check_serializing_named_tensor, is_ellipsis as is_ellipsis, resolve_ellipsis as resolve_ellipsis, single_ellipsis_index as single_ellipsis_index, unzip_namedshape as unzip_namedshape, update_names as update_names
from torch.overrides import get_default_nowrap_functions as get_default_nowrap_functions, handle_torch_function as handle_torch_function, has_torch_function as has_torch_function, has_torch_function_unary as has_torch_function_unary, has_torch_function_variadic as has_torch_function_variadic
from typing import Any

def _handle_torch_function_and_wrap_type_error_to_not_implemented(f): ...
def _rebuild_from_type(func, type, args, dict): ...
def _rebuild_from_type_v2(func, new_type, args, state): ...
def _dtype_to_typestr(dtype): ...

class Tensor(torch._C.TensorBase):
    _is_param: bool
    def _clear_non_serializable_cached_data(self):
        """Clears any data cached in the tensor's ``__dict__`` that would prevent the tensor
        from being serialized.

        For example, subclasses with custom dispatched sizes / strides cache this info in
        non-serializable PyCapsules within the ``__dict__``, and this must be cleared out for
        serialization to function.

        Any subclass that overrides this MUST call ``super()._clear_non_serializable_cached_data().``
        Additional data cleared within the override must be able to be re-cached transparently
        to avoid breaking subclass functionality.
        """
    def __deepcopy__(self, memo): ...
    def __reduce_ex__(self, proto): ...
    def storage(self):
        """
        storage() -> torch.TypedStorage

        Returns the underlying :class:`TypedStorage`.

        .. warning::

            :class:`TypedStorage` is deprecated. It will be removed in the future, and
            :class:`UntypedStorage` will be the only storage class. To access the
            :class:`UntypedStorage` directly, use :attr:`Tensor.untyped_storage()`.
        """
    def _typed_storage(self): ...
    def _reduce_ex_internal(self, proto): ...
    data: Incomplete
    def __setstate__(self, state): ...
    def __repr__(self, *, tensor_contents=None) -> str: ...
    def backward(self, gradient=None, retain_graph=None, create_graph: bool = False, inputs=None):
        """Computes the gradient of current tensor wrt graph leaves.

        The graph is differentiated using the chain rule. If the tensor is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionally requires specifying a ``gradient``.
        It should be a tensor of matching type and shape, that represents
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to zero
        ``.grad`` attributes or set them to ``None`` before calling it.
        See :ref:`Default gradient layouts<default-grad-layouts>`
        for details on the memory layout of accumulated gradients.

        .. note::

            If you run any forward ops, create ``gradient``, and/or call ``backward``
            in a user-specified CUDA stream context, see
            :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

        .. note::

            When ``inputs`` are provided and a given input is not a leaf,
            the current implementation will call its grad_fn (though it is not strictly needed to get this gradients).
            It is an implementation detail on which the user should not rely.
            See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

        Args:
            gradient (Tensor, optional): The gradient of the function
                being differentiated w.r.t. ``self``.
                This argument can be omitted if ``self`` is a scalar. Defaults to ``None``.
            retain_graph (bool, optional): If ``False``, the graph used to compute the grads will be freed;
                If ``True``, it will be retained. The default is ``None``, in which case the value is inferred from ``create_graph``
                (i.e., the graph is retained only when higher-order derivative tracking is requested). Note that in nearly all cases
                setting this option to True is not needed and often can be worked around in a much more efficient way.
            create_graph (bool, optional): If ``True``, graph of the derivative will
                be constructed, allowing to compute higher order derivative
                products. Defaults to ``False``.
            inputs (Sequence[Tensor], optional): Inputs w.r.t. which the gradient will be
                accumulated into ``.grad``. All other tensors will be ignored. If not
                provided, the gradient is accumulated into all the leaf Tensors that were
                used to compute the :attr:`tensors`. Defaults to ``None``.
        """
    _backward_hooks: Incomplete
    def register_hook(self, hook):
        """Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        Tensor is computed. The hook should have the following signature::

            hook(grad) -> Tensor or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v.grad

             2
             4
             6
            [torch.FloatTensor of size (3,)]

            >>> h.remove()  # removes the hook
        """
    _post_accumulate_grad_hooks: dict[Any, Any]
    def register_post_accumulate_grad_hook(self, hook):
        """Registers a backward hook that runs after grad accumulation.

        The hook will be called after all gradients for a tensor have been accumulated,
        meaning that the .grad field has been updated on that tensor. The post
        accumulate grad hook is ONLY applicable for leaf tensors (tensors without a
        .grad_fn field). Registering this hook on a non-leaf tensor will error!

        The hook should have the following signature::

            hook(param: Tensor) -> None

        Note that, unlike other autograd hooks, this hook operates on the tensor
        that requires grad and not the grad itself. The hook can in-place modify
        and access its Tensor argument, including its .grad field.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks. Since
            this hook runs during the backward pass, it will run in no_grad mode (unless
            create_graph is True). You can use torch.enable_grad() to re-enable autograd
            within the hook if you need it.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> lr = 0.01
            >>> # simulate a simple SGD update
            >>> h = v.register_post_accumulate_grad_hook(lambda p: p.add_(p.grad, alpha=-lr))
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v
            tensor([-0.0100, -0.0200, -0.0300], requires_grad=True)

            >>> h.remove()  # removes the hook
        """
    def reinforce(self, reward): ...
    detach: Incomplete
    detach_: Incomplete
    def is_shared(self):
        """Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
    def share_memory_(self):
        """Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.

        See :meth:`torch.UntypedStorage.share_memory_` for more details.
        """
    def module_load(self, other, assign: bool = False):
        """Defines how to transform ``other`` when loading it into ``self`` in :meth:`~nn.Module.load_state_dict`.

        Used when :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        It is expected that ``self`` is a parameter or buffer in an ``nn.Module`` and ``other`` is the
        value in the state dictionary with the corresponding key, this method defines
        how ``other`` is remapped before being swapped with ``self`` via
        :func:`~torch.utils.swap_tensors` in :meth:`~nn.Module.load_state_dict`.

        .. note::
            This method should always return a new object that is not ``self`` or ``other``.
            For example, the default implementation returns ``self.copy_(other).detach()``
            if ``assign`` is ``False`` or ``other.detach()`` if ``assign`` is ``True``.

        Args:
            other (Tensor): value in state dict with key corresponding to ``self``
            assign (bool): the assign argument passed to :meth:`nn.Module.load_state_dict`

        """
    def __reversed__(self):
        """Reverses the tensor along dimension 0."""
    def norm(self, p: float | str | None = 'fro', dim=None, keepdim: bool = False, dtype=None):
        """See :func:`torch.norm`"""
    def solve(self, other): ...
    def lstsq(self, other): ...
    def eig(self, eigenvectors: bool = False): ...
    def symeig(self, eigenvectors: bool = False): ...
    def lu(self, pivot: bool = True, get_infos: bool = False):
        """See :func:`torch.lu`"""
    def stft(self, n_fft: int, hop_length: int | None = None, win_length: int | None = None, window: Tensor | None = None, center: bool = True, pad_mode: str = 'reflect', normalized: bool = False, onesided: bool | None = None, return_complex: bool | None = None, align_to_window: bool | None = None):
        """See :func:`torch.stft`

        .. warning::
          This function changed signature at version 0.4.1. Calling with
          the previous signature may cause error or return incorrect result.
        """
    def istft(self, n_fft: int, hop_length: int | None = None, win_length: int | None = None, window: Tensor | None = None, center: bool = True, normalized: bool = False, onesided: bool | None = None, length: int | None = None, return_complex: bool = False):
        """See :func:`torch.istft`"""
    def resize(self, *sizes): ...
    def resize_as(self, tensor): ...
    def split(self, split_size, dim: int = 0):
        """See :func:`torch.split`"""
    def unique(self, sorted: bool = True, return_inverse: bool = False, return_counts: bool = False, dim=None):
        """Returns the unique elements of the input tensor.

        See :func:`torch.unique`
        """
    def unique_consecutive(self, return_inverse: bool = False, return_counts: bool = False, dim=None):
        """Eliminates all but the first element from every consecutive group of equivalent elements.

        See :func:`torch.unique_consecutive`
        """
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rsub__(self, other): ...
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rdiv__(self, other): ...
    __rtruediv__ = __rdiv__
    __itruediv__: Incomplete
    __pow__: Incomplete
    __ipow__: Incomplete
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmod__(self, other): ...
    def __format__(self, format_spec) -> str: ...
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rpow__(self, other): ...
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __floordiv__(self, other): ...
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rfloordiv__(self, other): ...
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rlshift__(self, other): ...
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rrshift__(self, other): ...
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmatmul__(self, other): ...
    __pos__: Incomplete
    __neg__: Incomplete
    __abs__: Incomplete
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __hash__(self): ...
    def __dir__(self): ...
    __array_priority__: int
    def __array__(self, dtype=None): ...
    def __array_wrap__(self, array): ...
    def __contains__(self, element: Any) -> bool:
        '''Check if `element` is present in tensor

        Args:
            element (Tensor or scalar): element to be checked
                for presence in current tensor"
        '''
    @property
    def __cuda_array_interface__(self):
        """Array view description for cuda tensors.

        See:
        https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
        """
    def storage_type(self):
        """storage_type() -> type

        Returns the type of the underlying storage.

        """
    def refine_names(self, *names):
        '''Refines the dimension names of :attr:`self` according to :attr:`names`.

        Refining is a special case of renaming that "lifts" unnamed dimensions.
        A ``None`` dim can be refined to have any name; a named dim can only be
        refined to have the same name.

        Because named tensors can coexist with unnamed tensors, refining names
        gives a nice way to write named-tensor-aware code that works with both
        named and unnamed tensors.

        :attr:`names` may contain up to one Ellipsis (``...``).
        The Ellipsis is expanded greedily; it is expanded in-place to fill
        :attr:`names` to the same length as ``self.dim()`` using names from the
        corresponding indices of ``self.names``.

        Python 2 does not support Ellipsis but one may use a string literal
        instead (``\'...\'``).

        Args:
            names (iterable of str): The desired names of the output tensor. May
                contain up to one Ellipsis.

        Examples::

            >>> imgs = torch.randn(32, 3, 128, 128)
            >>> named_imgs = imgs.refine_names(\'N\', \'C\', \'H\', \'W\')
            >>> named_imgs.names
            (\'N\', \'C\', \'H\', \'W\')

            >>> tensor = torch.randn(2, 3, 5, 7, 11)
            >>> tensor = tensor.refine_names(\'A\', ..., \'B\', \'C\')
            >>> tensor.names
            (\'A\', None, None, \'B\', \'C\')

        .. warning::
            The named tensor API is experimental and subject to change.

        '''
    def align_to(self, *names):
        """Permutes the dimensions of the :attr:`self` tensor to match the order
        specified in :attr:`names`, adding size-one dims for any new names.

        All of the dims of :attr:`self` must be named in order to use this method.
        The resulting tensor is a view on the original tensor.

        All dimension names of :attr:`self` must be present in :attr:`names`.
        :attr:`names` may contain additional names that are not in ``self.names``;
        the output tensor has a size-one dimension for each of those new names.

        :attr:`names` may contain up to one Ellipsis (``...``).
        The Ellipsis is expanded to be equal to all dimension names of :attr:`self`
        that are not mentioned in :attr:`names`, in the order that they appear
        in :attr:`self`.

        Python 2 does not support Ellipsis but one may use a string literal
        instead (``'...'``).

        Args:
            names (iterable of str): The desired dimension ordering of the
                output tensor. May contain up to one Ellipsis that is expanded
                to all unmentioned dim names of :attr:`self`.

        Examples::

            >>> tensor = torch.randn(2, 2, 2, 2, 2, 2)
            >>> named_tensor = tensor.refine_names('A', 'B', 'C', 'D', 'E', 'F')

            # Move the F and E dims to the front while keeping the rest in order
            >>> named_tensor.align_to('F', 'E', ...)

        .. warning::
            The named tensor API is experimental and subject to change.

        """
    def unflatten(self, dim, sizes):
        """
        unflatten(dim, sizes) -> Tensor

        See :func:`torch.unflatten`.

        """
    def rename_(self, *names, **rename_map):
        """In-place version of :meth:`~Tensor.rename`."""
    def rename(self, *names, **rename_map):
        """Renames dimension names of :attr:`self`.

        There are two main usages:

        ``self.rename(**rename_map)`` returns a view on tensor that has dims
        renamed as specified in the mapping :attr:`rename_map`.

        ``self.rename(*names)`` returns a view on tensor, renaming all
        dimensions positionally using :attr:`names`.
        Use ``self.rename(None)`` to drop names on a tensor.

        One cannot specify both positional args :attr:`names` and keyword args
        :attr:`rename_map`.

        Examples::

            >>> imgs = torch.rand(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
            >>> renamed_imgs = imgs.rename(N='batch', C='channels')
            >>> renamed_imgs.names
            ('batch', 'channels', 'H', 'W')

            >>> renamed_imgs = imgs.rename(None)
            >>> renamed_imgs.names
            (None, None, None, None)

            >>> renamed_imgs = imgs.rename('batch', 'channel', 'height', 'width')
            >>> renamed_imgs.names
            ('batch', 'channel', 'height', 'width')

        .. warning::
            The named tensor API is experimental and subject to change.

        """
    def to_sparse_coo(self):
        """Convert a tensor to :ref:`coordinate format <sparse-coo-docs>`.

        Examples::

             >>> dense = torch.randn(5, 5)
             >>> sparse = dense.to_sparse_coo()
             >>> sparse._nnz()
             25

        """
    def dim_order(self, *, ambiguity_check: bool | list[torch.memory_format] = False):
        '''
        dim_order(ambiguity_check=False) -> tuple

        Returns the uniquely determined tuple of int describing the dim order or
        physical layout of :attr:`self`.

        The dim order represents how dimensions are laid out in memory of dense tensors,
        starting from the outermost to the innermost dimension.

        Note that the dim order may not always be uniquely determined.
        If `ambiguity_check` is True, this function raises a RuntimeError when the dim order cannot be uniquely determined;
        If `ambiguity_check` is a list of memory formats, this function raises a RuntimeError when tensor can not be interpreted
        into exactly one of the given memory formats, or it cannot be uniquely determined.
        If `ambiguity_check` is False, it will return one of legal dim order(s) without checking its uniqueness.
        Otherwise, it will raise TypeError.

        Args:
            ambiguity_check (bool or List[torch.memory_format]): The check method for ambiguity of dim order.

        Examples::

            >>> torch.empty((2, 3, 5, 7)).dim_order()
            (0, 1, 2, 3)
            >>> torch.empty((2, 3, 5, 7)).transpose(1, 2).dim_order()
            (0, 2, 1, 3)
            >>> torch.empty((2, 3, 5, 7), memory_format=torch.channels_last).dim_order()
            (0, 2, 3, 1)
            >>> torch.empty((1, 2, 3, 4)).dim_order()
            (0, 1, 2, 3)
            >>> try:
            ...     torch.empty((1, 2, 3, 4)).dim_order(ambiguity_check=True)
            ... except RuntimeError as e:
            ...     print(e)
            The tensor does not have unique dim order, or cannot map to exact one of the given memory formats.
            >>> torch.empty((1, 2, 3, 4)).dim_order(
            ...     ambiguity_check=[torch.contiguous_format, torch.channels_last]
            ... )  # It can be mapped to contiguous format
            (0, 1, 2, 3)
            >>> try:
            ...     torch.empty((1, 2, 3, 4)).dim_order(ambiguity_check="ILLEGAL")
            ... except TypeError as e:
            ...     print(e)
            The ambiguity_check argument must be a bool or a list of memory formats.

        .. warning::
            The dim_order tensor API is experimental and subject to change.
        '''
    def _update_names(self, names, inplace): ...
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.

        One corollary to this is that you need coverage for torch.Tensor
        methods if implementing __torch_function__ for subclasses.

        We recommend always calling ``super().__torch_function__`` as the base
        case when doing the above.

        While not mandatory, we recommend making `__torch_function__` a classmethod.
        """
    __torch_dispatch__ = _C._disabled_torch_dispatch_impl
    def __dlpack__(self, stream=None):
        """
        Creates a DLpack `capsule https://data-apis.org/array-api/latest/design_topics/data_interchange.html#data-interchange`_
        of the current tensor to be exported to other libraries.

        This function will be called from the `from_dlpack` method
        of the library that will consume the capsule. `from_dlpack` passes the current
        stream to this method as part of the specification.

        Args:
            stream (integer or None): An optional Python integer representing a
            pointer to a CUDA stream. The current stream is synchronized with
            this stream before the capsule is created, and since the capsule
            shares its storage with the tensor this make it safe to access from
            both streams.  If None or -1 is passed then no synchronization is performed.
            If 1 (on CUDA) or 0 (on ROCM) then the default stream is used for
            synchronization.
        """
    def __dlpack_device__(self) -> tuple[enum.IntEnum, int]: ...
    __module__: str

def _convert(ret, cls): ...
