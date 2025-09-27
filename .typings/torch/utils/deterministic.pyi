import types

class _Deterministic(types.ModuleType):
    @property
    def fill_uninitialized_memory(self):
        """
        Whether to fill uninitialized memory with a known value when
        :meth:`torch.use_deterministic_algorithms()` is set to ``True``.
        """
    @fill_uninitialized_memory.setter
    def fill_uninitialized_memory(self, mode): ...
