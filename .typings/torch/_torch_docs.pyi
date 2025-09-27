from _typeshed import Incomplete

def parse_kwargs(desc):
    """Map a description of args to a dictionary of {argname: description}.

    Input:
        ('    weight (Tensor): a weight tensor\\n' +
         '        Some optional description')
    Output: {
        'weight': \\\n        'weight (Tensor): a weight tensor\\n        Some optional description'
    }
    """
def merge_dicts(*dicts):
    """Merge dictionaries into a single dictionary."""

common_args: Incomplete
reduceops_common_args: Incomplete
multi_dim_common: Incomplete
single_dim_common: Incomplete
factory_common_args: Incomplete
factory_like_common_args: Incomplete
factory_data_common_args: Incomplete
tf32_notes: Incomplete
rocm_fp16_notes: Incomplete
reproducibility_notes: dict[str, str]
sparse_support_notes: Incomplete
unary_foreach_func_name: Incomplete
unary_inplace_foreach_func_name: Incomplete
