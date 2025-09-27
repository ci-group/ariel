__all__ = ['fuse_known_modules', 'fuse_modules', 'fuse_modules_qat']

def fuse_known_modules(mod_list, is_qat, additional_fuser_method_mapping=None):
    """Return a list of known fuse modules.

    Returns a list of modules that fuses the operations specified
     in the input module list.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, bn
    linear, relu
    For these sequences, the first element in the output module list performs
    the fused operation. The rest of the elements are set to nn.Identity()
    """
def fuse_modules(model, modules_to_fuse, inplace: bool = False, fuser_func=..., fuse_custom_config_dict=None):
    '''Fuse a list of modules into a single module.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, relu
    bn, relu
    All other sequences are left unchanged.
    For these sequences, replaces the first item in the list
    with the fused module, replacing the rest of the modules
    with identity.

    Args:
        model: Model containing the modules to be fused
        modules_to_fuse: list of list of module names to fuse. Can also be a list
                         of strings if there is only a single list of modules to fuse.
        inplace: bool specifying if fusion happens in place on the model, by default
                 a new model is returned
        fuser_func: Function that takes in a list of modules and outputs a list of fused modules
                    of the same length. For example,
                    fuser_func([convModule, BNModule]) returns the list [ConvBNModule, nn.Identity()]
                    Defaults to torch.ao.quantization.fuse_known_modules
        `fuse_custom_config_dict`: custom configuration for fusion

    .. code-block:: python

       # Example of fuse_custom_config_dict
       fuse_custom_config_dict = {
           # Additional fuser_method mapping
           "additional_fuser_method_mapping": {
               (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn
           },
       }

    Returns:
        model with fused modules. A new copy is created if inplace=True.

    Examples::

            >>> # xdoctest: +SKIP
            >>> m = M().eval()
            >>> # m is a module containing the sub-modules below
            >>> modules_to_fuse = [ [\'conv1\', \'bn1\', \'relu1\'], [\'submodule.conv\', \'submodule.relu\']]
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

            >>> m = M().eval()
            >>> # Alternately provide a single list of modules to fuse
            >>> modules_to_fuse = [\'conv1\', \'bn1\', \'relu1\']
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

    '''
def fuse_modules_qat(model, modules_to_fuse, inplace: bool = False, fuser_func=..., fuse_custom_config_dict=None):
    """QAT version for `fuse_modules`."""
