import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod

class BasePruningMethod(ABC, metaclass=abc.ABCMeta):
    """Abstract base class for creation of new pruning techniques.

    Provides a skeleton for customization requiring the overriding of methods
    such as :meth:`compute_mask` and :meth:`apply`.
    """
    _tensor_name: str
    def __call__(self, module, inputs) -> None:
        """Multiply the mask into original tensor and store the result.

        Multiplies the mask (stored in ``module[name + '_mask']``)
        into the original tensor (stored in ``module[name + '_orig']``)
        and stores the result into ``module[name]`` by using :meth:`apply_mask`.

        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
    @abstractmethod
    def compute_mask(self, t, default_mask):
        """Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` according to the specific pruning
        method recipe.

        Args:
            t (torch.Tensor): tensor representing the importance scores of the
            parameter to prune.
            default_mask (torch.Tensor): Base mask from previous pruning
            iterations, that need to be respected after the new mask is
            applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``
        """
    def apply_mask(self, module):
        """Simply handles the multiplication between the parameter being pruned and the generated mask.

        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.

        Args:
            module (nn.Module): module containing the tensor to prune

        Returns:
            pruned_tensor (torch.Tensor): pruned version of the input tensor
        """
    @classmethod
    def apply(cls, module, name, *args, importance_scores=None, **kwargs):
        """Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            args: arguments passed on to a subclass of
                :class:`BasePruningMethod`
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the
                corresponding elements in the parameter being pruned.
                If unspecified or None, the parameter will be used in its place.
            kwargs: keyword arguments passed on to a subclass of a
                :class:`BasePruningMethod`
        """
    def prune(self, t, default_mask=None, importance_scores=None):
        """Compute and returns a pruned version of input tensor ``t``.

        According to the pruning rule specified in :meth:`compute_mask`.

        Args:
            t (torch.Tensor): tensor to prune (of same dimensions as
                ``default_mask``).
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as ``t``) used to compute mask for pruning ``t``.
                The values in this tensor indicate the importance of the
                corresponding elements in the ``t`` that is being pruned.
                If unspecified or None, the tensor ``t`` will be used in its place.
            default_mask (torch.Tensor, optional): mask from previous pruning
                iteration, if any. To be considered when determining what
                portion of the tensor that pruning should act on. If None,
                default to a mask of ones.

        Returns:
            pruned version of tensor ``t``.
        """
    def remove(self, module) -> None:
        """Remove the pruning reparameterization from a module.

        The pruned parameter named ``name`` remains permanently pruned,
        and the parameter named ``name+'_orig'`` is removed from the parameter list.
        Similarly, the buffer named ``name+'_mask'`` is removed from the buffers.

        Note:
            Pruning itself is NOT undone or reversed!
        """

class PruningContainer(BasePruningMethod):
    """Container holding a sequence of pruning methods for iterative pruning.

    Keeps track of the order in which pruning methods are applied and handles
    combining successive pruning calls.

    Accepts as argument an instance of a BasePruningMethod or an iterable of
    them.
    """
    _pruning_methods: tuple[BasePruningMethod, ...]
    _tensor_name: Incomplete
    def __init__(self, *args) -> None: ...
    def add_pruning_method(self, method) -> None:
        """Add a child pruning ``method`` to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, idx): ...
    def compute_mask(self, t, default_mask):
        """Apply the latest ``method`` by computing the new partial masks and returning its combination with the ``default_mask``.

        The new partial mask should be computed on the entries or channels
        that were not zeroed out by the ``default_mask``.
        Which portions of the tensor ``t`` the new mask will be calculated from
        depends on the ``PRUNING_TYPE`` (handled by the type handler):

        * for 'unstructured', the mask will be computed from the raveled
          list of nonmasked entries;

        * for 'structured', the mask will be computed from the nonmasked
          channels in the tensor;

        * for 'global', the mask will be computed across all entries.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
                (of same dimensions as ``default_mask``).
            default_mask (torch.Tensor): mask from previous pruning iteration.

        Returns:
            mask (torch.Tensor): new mask that combines the effects
            of the ``default_mask`` and the new mask from the current
            pruning ``method`` (of same dimensions as ``default_mask`` and
            ``t``).
        """

class Identity(BasePruningMethod):
    """Utility pruning method that does not prune any units but generates the pruning parametrization with a mask of ones."""
    PRUNING_TYPE: str
    def compute_mask(self, t, default_mask): ...
    @classmethod
    def apply(cls, module, name):
        """Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """

class RandomUnstructured(BasePruningMethod):
    """Prune (currently unpruned) units in a tensor at random.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """
    PRUNING_TYPE: str
    amount: Incomplete
    def __init__(self, amount) -> None: ...
    def compute_mask(self, t, default_mask): ...
    @classmethod
    def apply(cls, module, name, amount):
        """Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """

class L1Unstructured(BasePruningMethod):
    """Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """
    PRUNING_TYPE: str
    amount: Incomplete
    def __init__(self, amount) -> None: ...
    def compute_mask(self, t, default_mask): ...
    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        """Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """

class RandomStructured(BasePruningMethod):
    """Prune entire (currently unpruned) channels in a tensor at random.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """
    PRUNING_TYPE: str
    amount: Incomplete
    dim: Incomplete
    def __init__(self, amount, dim: int = -1) -> None: ...
    def compute_mask(self, t, default_mask):
        """Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` by randomly zeroing out channels
        along the specified dim of the tensor.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
    @classmethod
    def apply(cls, module, name, amount, dim: int = -1):
        """Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            dim (int, optional): index of the dim along which we define
                channels to prune. Default: -1.
        """

class LnStructured(BasePruningMethod):
    """Prune entire (currently unpruned) channels in a tensor based on their L\\ ``n``-norm.

    Args:
        amount (int or float): quantity of channels to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """
    PRUNING_TYPE: str
    amount: Incomplete
    n: Incomplete
    dim: Incomplete
    def __init__(self, amount, n, dim: int = -1) -> None: ...
    def compute_mask(self, t, default_mask):
        """Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a mask to apply on
        top of the ``default_mask`` by zeroing out the channels along the
        specified dim with the lowest L\\ ``n``-norm.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied.  Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
    @classmethod
    def apply(cls, module, name, amount, n, dim, importance_scores=None):
        """Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
                entries for argument ``p`` in :func:`torch.norm`.
            dim (int): index of the dim along which we define channels to
                prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """

class CustomFromMask(BasePruningMethod):
    PRUNING_TYPE: str
    mask: Incomplete
    def __init__(self, mask) -> None: ...
    def compute_mask(self, t, default_mask): ...
    @classmethod
    def apply(cls, module, name, mask):
        """Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """

def identity(module, name):
    '''Apply pruning reparametrization without pruning any units.

    Applies pruning reparametrization to the tensor corresponding to the
    parameter called ``name`` in ``module`` without actually pruning any
    units. Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+\'_mask\'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+\'_orig\'``.

    Note:
        The mask is a tensor of ones.

    Args:
        module (nn.Module): module containing the tensor to prune.
        name (str): parameter name within ``module`` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.identity(nn.Linear(2, 3), "bias")
        >>> print(m.bias_mask)
        tensor([1., 1., 1.])
    '''
def random_unstructured(module, name, amount):
    '''Prune tensor by removing random (currently unpruned) units.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) units
    selected at random.
    Modifies module in place (and also return the modified module) by:

    1) adding a named buffer called ``name+\'_mask\'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+\'_orig\'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.random_unstructured(nn.Linear(2, 3), "weight", amount=1)
        >>> torch.sum(m.weight_mask == 0)
        tensor(1)

    '''
def l1_unstructured(module, name, amount, importance_scores=None):
    '''Prune tensor by removing units with the lowest L1-norm.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified `amount` of (currently unpruned) units with the
    lowest L1-norm.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+\'_mask\'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+\'_orig\'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.l1_unstructured(nn.Linear(2, 3), "weight", amount=0.2)
        >>> m.state_dict().keys()
        odict_keys([\'bias\', \'weight_orig\', \'weight_mask\'])
    '''
def random_structured(module, name, amount, dim):
    '''Prune tensor by removing random channels along the specified dimension.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) channels
    along the specified ``dim`` selected at random.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+\'_mask\'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+\'_orig\'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int): index of the dim along which we define channels to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.random_structured(nn.Linear(5, 3), "weight", amount=3, dim=1)
        >>> columns_pruned = int(sum(torch.sum(m.weight, dim=0) == 0))
        >>> print(columns_pruned)
        3
    '''
def ln_structured(module, name, amount, n, dim, importance_scores=None):
    '''Prune tensor by removing channels with the lowest L\\ ``n``-norm along the specified dimension.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) channels
    along the specified ``dim`` with the lowest L\\ ``n``-norm.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+\'_mask\'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+\'_orig\'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, \'fro\', \'nuc\'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int): index of the dim along which we define channels to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = prune.ln_structured(
        ...     nn.Conv2d(5, 3, 2), "weight", amount=0.3, dim=1, n=float("-inf")
        ... )
    '''
def global_unstructured(parameters, pruning_method, importance_scores=None, **kwargs) -> None:
    '''
    Globally prunes tensors corresponding to all parameters in ``parameters`` by applying the specified ``pruning_method``.

    Modifies modules in place by:

    1) adding a named buffer called ``name+\'_mask\'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+\'_orig\'``.

    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE=\'unstructured\'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter\'s importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Raises:
        TypeError: if ``PRUNING_TYPE != \'unstructured\'``

    Note:
        Since global structured pruning doesn\'t make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.

    Examples:
        >>> from torch.nn.utils import prune
        >>> from collections import OrderedDict
        >>> net = nn.Sequential(
        ...     OrderedDict(
        ...         [
        ...             ("first", nn.Linear(10, 4)),
        ...             ("second", nn.Linear(4, 1)),
        ...         ]
        ...     )
        ... )
        >>> parameters_to_prune = (
        ...     (net.first, "weight"),
        ...     (net.second, "weight"),
        ... )
        >>> prune.global_unstructured(
        ...     parameters_to_prune,
        ...     pruning_method=prune.L1Unstructured,
        ...     amount=10,
        ... )
        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        tensor(10)

    '''
def custom_from_mask(module, name, mask):
    '''Prune tensor corresponding to parameter called ``name`` in ``module`` by applying the pre-computed mask in ``mask``.

    Modifies module in place (and also return the modified module) by:

    1) adding a named buffer called ``name+\'_mask\'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+\'_orig\'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.
        mask (Tensor): binary mask to be applied to the parameter.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = prune.custom_from_mask(
        ...     nn.Linear(5, 3), name="bias", mask=torch.tensor([0, 1, 0])
        ... )
        >>> print(m.bias_mask)
        tensor([0., 1., 0.])

    '''
def remove(module, name):
    '''Remove the pruning reparameterization from a module and the pruning method from the forward hook.

    The pruned parameter named ``name`` remains permanently pruned, and the parameter
    named ``name+\'_orig\'`` is removed from the parameter list. Similarly,
    the buffer named ``name+\'_mask\'`` is removed from the buffers.

    Note:
        Pruning itself is NOT undone or reversed!

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.

    Examples:
        >>> m = random_unstructured(nn.Linear(5, 7), name="weight", amount=0.2)
        >>> m = remove(m, name="weight")
    '''
def is_pruned(module):
    '''Check if a module is pruned by looking for pruning pre-hooks.

    Check whether ``module`` is pruned by looking for
    ``forward_pre_hooks`` in its modules that inherit from the
    :class:`BasePruningMethod`.

    Args:
        module (nn.Module): object that is either pruned or unpruned

    Returns:
        binary answer to whether ``module`` is pruned.

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = nn.Linear(5, 7)
        >>> print(prune.is_pruned(m))
        False
        >>> prune.random_unstructured(m, name="weight", amount=0.2)
        >>> print(prune.is_pruned(m))
        True
    '''
def _validate_pruning_amount_init(amount) -> None:
    """Validate helper to check the range of amount at init.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.

    Raises:
        ValueError: if amount is a float not in [0, 1], or if it's a negative
            integer.
        TypeError: if amount is neither a float nor an integer.

    Note:
        This does not take into account the number of parameters in the
        tensor to be pruned, which is known only at prune.
    """
def _validate_pruning_amount(amount, tensor_size) -> None:
    """Validate that the pruning amount is meaningful wrt to the size of the data.

    Validation helper to check that the amount of parameters to prune
    is meaningful wrt to the size of the data (`tensor_size`).

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.
    """
def _validate_structured_pruning(t) -> None:
    '''Validate that the tensor to be pruned is at least 2-Dimensional.

    Validation helper to check that the tensor to be pruned is multi-
    dimensional, such that the concept of "channels" is well-defined.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune

    Raises:
        ValueError: if the tensor `t` is not at least 2D.
    '''
def _compute_nparams_toprune(amount, tensor_size):
    """Convert the pruning amount from a percentage to absolute value.

    Since amount can be expressed either in absolute value or as a
    percentage of the number of units/channels in a tensor, this utility
    function converts the percentage to absolute value to standardize
    the handling of pruning.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.

    Returns:
        int: the number of units to prune in the tensor
    """
def _validate_pruning_dim(t, dim) -> None:
    """Validate that the pruning dimension is within the bounds of the tensor dimension.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        dim (int): index of the dim along which we define channels to prune
    """
def _compute_norm(t, n, dim):
    """Compute the L_n-norm of a tensor along all dimensions except for the specified dimension.

    The L_n-norm will be computed across all entries in tensor `t` along all dimension
    except for the one identified by dim.
    Example: if `t` is of shape, say, 3x2x4 and dim=2 (the last dim),
    then norm will have Size [4], and each entry will represent the
    `L_n`-norm computed using the 3x2=6 entries for each of the 4 channels.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument p in torch.norm
        dim (int): dim identifying the channels to prune

    Returns:
        norm (torch.Tensor): L_n norm computed across all dimensions except
            for `dim`. By construction, `norm.shape = t.shape[-1]`.
    """
