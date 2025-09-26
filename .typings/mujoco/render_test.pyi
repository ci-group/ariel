from _typeshed import Incomplete
from absl.testing import absltest

class MuJoCoRenderTest(absltest.TestCase):
    gl: Incomplete
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    model: Incomplete
    data: Incomplete
    def test_can_render(self) -> None:
        """Test that the bindings can successfully render a simple image.

    This test sets up a basic MuJoCo rendering context similar to the example in
    https://mujoco.readthedocs.io/en/latest/programming#visualization
    It calls `mjr_rectangle` rather than `mjr_render` so that we can assert an
    exact rendered image without needing golden data. The purpose of this test
    is to ensure that the bindings can correctly return pixels in Python, rather
    than to test MuJoCo's rendering pipeline itself.
    """
    def test_safe_to_free_context_twice(self) -> None: ...
    def test_mjrrect_repr(self) -> None: ...
