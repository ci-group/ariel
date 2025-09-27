import mujoco._structs
import numpy
from typing import overload

class MjrContext:
    auxColor: numpy.ndarray[numpy.uint32]
    auxColor_r: numpy.ndarray[numpy.uint32]
    auxFBO: numpy.ndarray[numpy.uint32]
    auxFBO_r: numpy.ndarray[numpy.uint32]
    auxHeight: numpy.ndarray[numpy.int32]
    auxSamples: numpy.ndarray[numpy.int32]
    auxWidth: numpy.ndarray[numpy.int32]
    baseBuiltin: int
    baseFontBig: int
    baseFontNormal: int
    baseFontShadow: int
    baseHField: int
    baseMesh: int
    basePlane: int
    charHeight: int
    charHeightBig: int
    charWidth: numpy.ndarray[numpy.int32]
    charWidthBig: numpy.ndarray[numpy.int32]
    currentBuffer: int
    fogEnd: float
    fogRGBA: numpy.ndarray[numpy.float32]
    fogStart: float
    fontScale: int
    glInitialized: int
    lineWidth: float
    mat_texid: numpy.ndarray[numpy.int32]
    mat_texrepeat: numpy.ndarray[numpy.float32]
    mat_texuniform: numpy.ndarray[numpy.int32]
    ntexture: int
    offColor: int
    offColor_r: int
    offDepthStencil: int
    offDepthStencil_r: int
    offFBO: int
    offFBO_r: int
    offHeight: int
    offSamples: int
    offWidth: int
    rangeBuiltin: int
    rangeFont: int
    rangeHField: int
    rangeMesh: int
    rangePlane: int
    readDepthMap: int
    readPixelFormat: int
    shadowClip: float
    shadowFBO: int
    shadowScale: float
    shadowSize: int
    shadowTex: int
    texture: numpy.ndarray[numpy.uint32]
    textureType: numpy.ndarray[numpy.int32]
    windowAvailable: int
    windowDoublebuffer: int
    windowSamples: int
    windowStereo: int
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._render.MjrContext) -> None

        2. __init__(self: mujoco._render.MjrContext, arg0: mujoco._structs.MjModel, arg1: int) -> None
        """
    @overload
    def __init__(self, arg0: mujoco._structs.MjModel, arg1: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._render.MjrContext) -> None

        2. __init__(self: mujoco._render.MjrContext, arg0: mujoco._structs.MjModel, arg1: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def free(self) -> None:
        """free(self: mujoco._render.MjrContext) -> None

        Frees resources in current active OpenGL context, sets struct to default.
        """
    @property
    def nskin(self) -> int: ...
    @property
    def skinfaceVBO(self) -> tuple: ...
    @property
    def skinnormalVBO(self) -> tuple: ...
    @property
    def skintexcoordVBO(self) -> tuple: ...
    @property
    def skinvertVBO(self) -> tuple: ...

class MjrRect:
    bottom: int
    height: int
    left: int
    width: int
    def __init__(self, left: int, bottom: int, width: int, height: int) -> None:
        """__init__(self: mujoco._render.MjrRect, left: int, bottom: int, width: int, height: int) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjrRect:
        """__copy__(self: mujoco._render.MjrRect) -> mujoco._render.MjrRect"""
    def __deepcopy__(self, arg0: dict) -> MjrRect:
        """__deepcopy__(self: mujoco._render.MjrRect, arg0: dict) -> mujoco._render.MjrRect"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

def mjr_addAux(index: int, width: int, height: int, samples: int, con: MjrContext) -> None:
    """mjr_addAux(index: int, width: int, height: int, samples: int, con: mujoco._render.MjrContext) -> None

    Add Aux buffer with given index to context; free previous Aux buffer.
    """
def mjr_blitAux(index: int, src: MjrRect, left: int, bottom: int, con: MjrContext) -> None:
    """mjr_blitAux(index: int, src: mujoco._render.MjrRect, left: int, bottom: int, con: mujoco._render.MjrContext) -> None

    Blit from Aux buffer to con->currentBuffer.
    """
def mjr_blitBuffer(src: MjrRect, dst: MjrRect, flg_color: int, flg_depth: int, con: MjrContext) -> None:
    """mjr_blitBuffer(src: mujoco._render.MjrRect, dst: mujoco._render.MjrRect, flg_color: int, flg_depth: int, con: mujoco._render.MjrContext) -> None

    Blit from src viewpoint in current framebuffer to dst viewport in other framebuffer. If src, dst have different size and flg_depth==0, color is interpolated with GL_LINEAR.
    """
def mjr_changeFont(fontscale: int, con: MjrContext) -> None:
    """mjr_changeFont(fontscale: int, con: mujoco._render.MjrContext) -> None

    Change font of existing context.
    """
def mjr_drawPixels(rgb: numpy.ndarray[numpy.uint8[m, 1]] | None, depth: numpy.ndarray[numpy.float32[m, 1]] | None, viewport: MjrRect, con: MjrContext) -> None:
    """mjr_drawPixels(rgb: Optional[numpy.ndarray[numpy.uint8[m, 1]]], depth: Optional[numpy.ndarray[numpy.float32[m, 1]]], viewport: mujoco._render.MjrRect, con: mujoco._render.MjrContext) -> None

    Draw pixels from client buffer to current OpenGL framebuffer. Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
    """
def mjr_figure(viewport: MjrRect, fig: mujoco._structs.MjvFigure, con: MjrContext) -> None:
    """mjr_figure(viewport: mujoco._render.MjrRect, fig: mujoco._structs.MjvFigure, con: mujoco._render.MjrContext) -> None

    Draw 2D figure.
    """
def mjr_findRect(x: int, y: int, nrect: int, rect: MjrRect) -> int:
    """mjr_findRect(x: int, y: int, nrect: int, rect: mujoco._render.MjrRect) -> int

    Find first rectangle containing mouse, -1: not found.
    """
def mjr_finish() -> None:
    """mjr_finish() -> None

    Call glFinish.
    """
def mjr_getError() -> int:
    """mjr_getError() -> int

    Call glGetError and return result.
    """
def mjr_label(viewport: MjrRect, font: int, txt: str, r: float, g: float, b: float, a: float, rt: float, gt: float, bt: float, con: MjrContext) -> None:
    """mjr_label(viewport: mujoco._render.MjrRect, font: int, txt: str, r: float, g: float, b: float, a: float, rt: float, gt: float, bt: float, con: mujoco._render.MjrContext) -> None

    Draw rectangle with centered text.
    """
def mjr_maxViewport(con: MjrContext) -> MjrRect:
    """mjr_maxViewport(con: mujoco._render.MjrContext) -> mujoco._render.MjrRect

    Get maximum viewport for active buffer.
    """
def mjr_overlay(font: int, gridpos: int, viewport: MjrRect, overlay: str, overlay2: str, con: MjrContext) -> None:
    """mjr_overlay(font: int, gridpos: int, viewport: mujoco._render.MjrRect, overlay: str, overlay2: str, con: mujoco._render.MjrContext) -> None

    Draw text overlay; font is mjtFont; gridpos is mjtGridPos.
    """
def mjr_readPixels(rgb: numpy.ndarray[numpy.uint8] | None, depth: numpy.ndarray[numpy.float32] | None, viewport: MjrRect, con: MjrContext) -> None:
    """mjr_readPixels(rgb: Optional[numpy.ndarray[numpy.uint8]], depth: Optional[numpy.ndarray[numpy.float32]], viewport: mujoco._render.MjrRect, con: mujoco._render.MjrContext) -> None

    Read pixels from current OpenGL framebuffer to client buffer. Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
    """
def mjr_rectangle(viewport: MjrRect, r: float, g: float, b: float, a: float) -> None:
    """mjr_rectangle(viewport: mujoco._render.MjrRect, r: float, g: float, b: float, a: float) -> None

    Draw rectangle.
    """
def mjr_render(viewport: MjrRect, scn: mujoco._structs.MjvScene, con: MjrContext) -> None:
    """mjr_render(viewport: mujoco._render.MjrRect, scn: mujoco._structs.MjvScene, con: mujoco._render.MjrContext) -> None

    Render 3D scene.
    """
def mjr_resizeOffscreen(width: int, height: int, con: MjrContext) -> None:
    """mjr_resizeOffscreen(width: int, height: int, con: mujoco._render.MjrContext) -> None

    Resize offscreen buffers.
    """
def mjr_restoreBuffer(con: MjrContext) -> None:
    """mjr_restoreBuffer(con: mujoco._render.MjrContext) -> None

    Make con->currentBuffer current again.
    """
def mjr_setAux(index: int, con: MjrContext) -> None:
    """mjr_setAux(index: int, con: mujoco._render.MjrContext) -> None

    Set Aux buffer for custom OpenGL rendering (call restoreBuffer when done).
    """
def mjr_setBuffer(framebuffer: int, con: MjrContext) -> None:
    """mjr_setBuffer(framebuffer: int, con: mujoco._render.MjrContext) -> None

    Set OpenGL framebuffer for rendering: mjFB_WINDOW or mjFB_OFFSCREEN. If only one buffer is available, set that buffer and ignore framebuffer argument.
    """
def mjr_text(font: int, txt: str, con: MjrContext, x: float, y: float, r: float, g: float, b: float) -> None:
    """mjr_text(font: int, txt: str, con: mujoco._render.MjrContext, x: float, y: float, r: float, g: float, b: float) -> None

    Draw text at (x,y) in relative coordinates; font is mjtFont.
    """
def mjr_uploadHField(m: mujoco._structs.MjModel, con: MjrContext, hfieldid: int) -> None:
    """mjr_uploadHField(m: mujoco._structs.MjModel, con: mujoco._render.MjrContext, hfieldid: int) -> None

    Upload height field to GPU, overwriting previous upload if any.
    """
def mjr_uploadMesh(m: mujoco._structs.MjModel, con: MjrContext, meshid: int) -> None:
    """mjr_uploadMesh(m: mujoco._structs.MjModel, con: mujoco._render.MjrContext, meshid: int) -> None

    Upload mesh to GPU, overwriting previous upload if any.
    """
def mjr_uploadTexture(m: mujoco._structs.MjModel, con: MjrContext, texid: int) -> None:
    """mjr_uploadTexture(m: mujoco._structs.MjModel, con: mujoco._render.MjrContext, texid: int) -> None

    Upload texture to GPU, overwriting previous upload if any.
    """
