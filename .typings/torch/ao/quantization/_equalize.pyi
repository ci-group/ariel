__all__ = ['set_module_weight', 'set_module_bias', 'has_bias', 'get_module_weight', 'get_module_bias', 'max_over_ndim', 'min_over_ndim', 'channel_range', 'get_name_by_module', 'cross_layer_equalization', 'process_paired_modules_list_to_name', 'expand_groups_in_paired_modules_list', 'equalize', 'converged']

def set_module_weight(module, weight) -> None: ...
def set_module_bias(module, bias) -> None: ...
def has_bias(module) -> bool: ...
def get_module_weight(module): ...
def get_module_bias(module): ...
def max_over_ndim(input, axis_list, keepdim: bool = False):
    """Apply 'torch.max' over the given axes."""
def min_over_ndim(input, axis_list, keepdim: bool = False):
    """Apply 'torch.min' over the given axes."""
def channel_range(input, axis: int = 0):
    """Find the range of weights associated with a specific channel."""
def get_name_by_module(model, module):
    """Get the name of a module within a model.

    Args:
        model: a model (nn.module) that equalization is to be applied on
        module: a module within the model

    Returns:
        name: the name of the module within the model
    """
def cross_layer_equalization(module1, module2, output_axis: int = 0, input_axis: int = 1) -> None:
    """Scale the range of Tensor1.output to equal Tensor2.input.

    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
def process_paired_modules_list_to_name(model, paired_modules_list):
    """Processes a list of paired modules to a list of names of paired modules."""
def expand_groups_in_paired_modules_list(paired_modules_list):
    """Expands module pair groups larger than two into groups of two modules."""
def equalize(model, paired_modules_list, threshold: float = 0.0001, inplace: bool = True):
    """Equalize modules until convergence is achieved.

    Given a list of adjacent modules within a model, equalization will
    be applied between each pair, this will repeated until convergence is achieved

    Keeps a copy of the changing modules from the previous iteration, if the copies
    are not that different than the current modules (determined by converged_test),
    then the modules have converged enough that further equalizing is not necessary

    Reference is section 4.1 of this paper https://arxiv.org/pdf/1906.04721.pdf

    Args:
        model: a model (nn.Module) that equalization is to be applied on
            paired_modules_list (List(List[nn.module || str])): a list of lists
            where each sublist is a pair of two submodules found in the model,
            for each pair the two modules have to be adjacent in the model,
            with only piece-wise-linear functions like a (P)ReLU or LeakyReLU in between
            to get expected results.
            The list can contain either modules, or names of modules in the model.
            If you pass multiple modules in the same list, they will all be equalized together.
            threshold (float): a number used by the converged function to determine what degree
            of similarity between models is necessary for them to be called equivalent
        inplace (bool): determines if function is inplace or not
    """
def converged(curr_modules, prev_modules, threshold: float = 0.0001):
    """Test whether modules are converged to a specified threshold.

    Tests for the summed norm of the differences between each set of modules
    being less than the given threshold

    Takes two dictionaries mapping names to modules, the set of names for each dictionary
    should be the same, looping over the set of names, for each name take the difference
    between the associated modules in each dictionary

    """
