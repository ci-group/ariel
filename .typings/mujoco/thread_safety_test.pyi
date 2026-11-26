from absl.testing import absltest

class ConcurrentStressTest(absltest.TestCase):
    """Tests concurrent access to bindings under multiple threads."""
    def test_concurrent_indexer_access(self) -> None:
        """Concurrent first-access of lazy indexers must not corrupt state."""
    def test_concurrent_struct_list_access(self) -> None:
        """Concurrent access to contact struct lists must not crash."""
    def test_concurrent_callback_set_get(self) -> None:
        """Concurrent callback setter/getter calls must not crash or deadlock."""
    def test_concurrent_model_data_lifecycle(self) -> None:
        """Concurrent MjModel/MjData creation and destruction must not crash."""

class ReentrancyTest(absltest.TestCase):
    """Tests that mutexes don't deadlock when __del__ re-enters setters."""
    def test_callback_setter_from_del(self) -> None:
        """Setting a callback whose __del__ re-enters the setter must not hang."""
    def test_callback_replacement_from_del(self) -> None:
        """Replacing a callback whose __del__ sets a new callback must not hang."""
