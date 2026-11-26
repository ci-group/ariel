from absl.testing import parameterized

class MuJoCoRendererTest(parameterized.TestCase):
    def test_renderer_info_binding(self) -> None: ...
    def test_renderer_unknown_camera_name(self) -> None: ...
    def test_renderer_camera_under_range(self) -> None: ...
    def test_renderer_camera_over_range(self) -> None: ...
    def test_renderer_renders_scene(self) -> None: ...
    def test_renderer_output_without_out(self) -> None: ...
    def test_renderer_output_with_out(self) -> None: ...
    def test_renderer_output_with_cast_and_non_contiguous_out(self) -> None: ...
    def test_renderer_restores_scene_flags_on_exception(self) -> None: ...
    def test_renderer_depth_rendering_camera_projection(self):
        """Tests orthographic vs perspective camera depth rendering projections.

    Orthographic depth uses a linear NDC mapping (glOrtho); perspective depth
    uses a hyperbolic mapping (glFrustum).

    Render the same tilted plane using identically positioned orthographic and
    perspective cameras. The two should agree on the depth at the center pixel
    (actual distance), but they should have different depth values at non-center
    pixels because the rays of the perspective camera diverge while those of the
    orthographic camera remain parallel.
    """
    def test_mjr_read_pixels_depth_map(self, camera_name) -> None:
        """Tests mjr_readPixels raw depth mapping and far-plane clipping."""
    def test_renderer_del_safe_when_init_fails_early(self):
        """Regression test for #3213.

    Renderer.__del__ must be safe on a partially-constructed instance when
    __init__ raises before the rendering contexts are assigned. Previously,
    AttributeError from __del__ masked the real __init__ exception.
    """
    def test_renderer_close_preserves_active_context(self) -> None:
        """Closing or GC-collecting a Renderer must not clobber another context."""
