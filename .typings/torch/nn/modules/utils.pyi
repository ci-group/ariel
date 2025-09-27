from typing import Any

__all__ = ['consume_prefix_in_state_dict_if_present']

def consume_prefix_in_state_dict_if_present(state_dict: dict[str, Any], prefix: str) -> None:
    '''Strip the prefix in state_dict in place, if any.

    .. note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    '''
