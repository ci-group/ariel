from absl.testing import absltest

class MemoryLeakTest(absltest.TestCase):
    def test_deepcopy_mjdata_leak(self) -> None: ...
    def _memory_limit(self, limit_in_bytes: int) -> int:
        """Limits max memory usage, and returns previous limit."""
