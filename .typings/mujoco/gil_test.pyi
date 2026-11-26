from absl.testing import absltest

class GilDisabledTest(absltest.TestCase):
    def test_gil_is_disabled(self) -> None: ...
