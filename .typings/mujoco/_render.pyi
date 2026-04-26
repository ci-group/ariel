import mujoco._structs
import numpy
import numpy.typing
import typing
from typing import overload

class MjrContext:
    auxColor: numpy.typing.NDArray[numpy.uint32]
    auxColor_r: numpy.typing.NDArray[numpy.uint32]
    auxFBO: numpy.typing.NDArray[numpy.uint32]
    auxFBO_r: numpy.typing.NDArray[numpy.uint32]
    auxHeight: numpy.typing.NDArray[numpy.int32]
    auxSamples: numpy.typing.NDArray[numpy.int32]
    auxWidth: numpy.typing.NDArray[numpy.int32]
    baseBuiltin: int
    baseFontBig: int
    baseFontNormal: int
    baseFontShadow: int
    baseHField: int
    baseMesh: int
    basePlane: int
    charHeight: int
    charHeightBig: int
    charWidth: numpy.typing.NDArray[numpy.int32]
    charWidthBig: numpy.typing.NDArray[numpy.int32]
    currentBuffer: int
    fogEnd: float
    fogRGBA: numpy.typing.NDArray[numpy.float32]
    fogStart: float
    fontScale: int
    glInitialized: int
    lineWidth: float
    mat_texid: numpy.typing.NDArray[numpy.int32]
    mat_texrepeat: numpy.typing.NDArray[numpy.float32]
    mat_texuniform: numpy.typing.NDArray[numpy.int32]
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
    texture: numpy.typing.NDArray[numpy.uint32]
    textureType: numpy.typing.NDArray[numpy.int32]
    windowAvailable: int
    windowDoublebuffer: int
    windowSamples: int
    windowStereo: int
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._render.MjrContext) -> None

        2. __init__(self: mujoco._render.MjrContext, arg0: mujoco._structs.MjModel, arg1: typing.SupportsInt) -> None
        """
    @overload
    def __init__(self, arg0: mujoco._structs.MjModel, arg1: typing.SupportsInt) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._render.MjrContext) -> None

        2. __init__(self: mujoco._render.MjrContext, arg0: mujoco._structs.MjModel, arg1: typing.SupportsInt) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def free(self) -> None:
        """free(self: mujoco._render.MjrContext) -> None

        Frees resources in current active OpenGL context, sets struct to default.
        """
    @property
    def nskin(self) -> int:
        """(arg0: mujoco._render.MjrContext) -> int"""
    @property
    def skinfaceVBO(self) -> tuple:
        """(arg0: mujoco._render.MjrContext) -> tuple"""
    @property
    def skinnormalVBO(self) -> tuple:
        """(arg0: mujoco._render.MjrContext) -> tuple"""
    @property
    def skintexcoordVBO(self) -> tuple:
        """(arg0: mujoco._render.MjrContext) -> tuple"""
    @property
    def skinvertVBO(self) -> tuple:
        """(arg0: mujoco._render.MjrContext) -> tuple"""

class MjrRect:
    bottom: int
    height: int
    left: int
    width: int
    def __init__(self, left: typing.SupportsInt, bottom: typing.SupportsInt, width: typing.SupportsInt, height: typing.SupportsInt) -> None:
        """__init__(self: mujoco._render.MjrRect, left: typing.SupportsInt, bottom: typing.SupportsInt, width: typing.SupportsInt, height: typing.SupportsInt) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjrRect:
        """__copy__(self: mujoco._render.MjrRect) -> mujoco._render.MjrRect"""
    def __deepcopy__(self, arg0: dict) -> MjrRect:
        """__deepcopy__(self: mujoco._render.MjrRect, arg0: dict) -> mujoco._render.MjrRect"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

def mjr_addAux(index: typing.SupportsInt, width: typing.SupportsInt, height: typing.SupportsInt, samples: typing.SupportsInt, con: MjrContext) -> None:
    """mjr_addAux(index: typing.SupportsInt, width: typing.SupportsInt, height: typing.SupportsInt, samples: typing.SupportsInt, con: mujoco._render.MjrContext) -> None

    Add Aux buffer with given index to context; free previous Aux buffer.
    """
def mjr_blitAux(index: typing.SupportsInt, src: MjrRect, left: typing.SupportsInt, bottom: typing.SupportsInt, con: MjrContext) -> None:
    """mjr_blitAux(index: typing.SupportsInt, src: mujoco._render.MjrRect, left: typing.SupportsInt, bottom: typing.SupportsInt, con: mujoco._render.MjrContext) -> None

    Blit from Aux buffer to con->currentBuffer.
    """
def mjr_blitBuffer(src: MjrRect, dst: MjrRect, flg_color: typing.SupportsInt, flg_depth: typing.SupportsInt, con: MjrContext) -> None:
    """mjr_blitBuffer(src: mujoco._render.MjrRect, dst: mujoco._render.MjrRect, flg_color: typing.SupportsInt, flg_depth: typing.SupportsInt, con: mujoco._render.MjrContext) -> None

    Blit from src viewpoint in current framebuffer to dst viewport in other framebuffer. If src, dst have different size and flg_depth==0, color is interpolated with GL_LINEAR.
    """
def mjr_changeFont(fontscale: typing.SupportsInt, con: MjrContext) -> None:
    """mjr_changeFont(fontscale: typing.SupportsInt, con: mujoco._render.MjrContext) -> None

    Change font of existing context.
    """
def mjr_drawPixels(rgb: typing.Annotated[numpy.typing.NDArray[numpy.uint8], '[m, 1]'] | None, depth: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[m, 1]'] | None, viewport: MjrRect, con: MjrContext) -> None:
    '''mjr_drawPixels(rgb: typing.Annotated[numpy.typing.NDArray[numpy.uint8], "[m, 1]"] | None, depth: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, 1]"] | None, viewport: mujoco._render.MjrRect, con: mujoco._render.MjrContext) -> None

    Draw pixels from client buffer to current OpenGL framebuffer. Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
    '''
def mjr_figure(viewport: MjrRect, fig: mujoco._structs.MjvFigure, con: MjrContext) -> None:
    """mjr_figure(viewport: mujoco._render.MjrRect, fig: mujoco._structs.MjvFigure, con: mujoco._render.MjrContext) -> None

    Draw 2D figure.
    """
def mjr_findRect(x: typing.SupportsInt, y: typing.SupportsInt, nrect: typing.SupportsInt, rect: MjrRect) -> int:
    """mjr_findRect(x: typing.SupportsInt, y: typing.SupportsInt, nrect: typing.SupportsInt, rect: mujoco._render.MjrRect) -> int

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
def mjr_label(viewport: MjrRect, font: typing.SupportsInt, txt: str, r: typing.SupportsFloat, g: typing.SupportsFloat, b: typing.SupportsFloat, a: typing.SupportsFloat, rt: typing.SupportsFloat, gt: typing.SupportsFloat, bt: typing.SupportsFloat, con: MjrContext) -> None:
    """mjr_label(viewport: mujoco._render.MjrRect, font: typing.SupportsInt, txt: str, r: typing.SupportsFloat, g: typing.SupportsFloat, b: typing.SupportsFloat, a: typing.SupportsFloat, rt: typing.SupportsFloat, gt: typing.SupportsFloat, bt: typing.SupportsFloat, con: mujoco._render.MjrContext) -> None

    Draw rectangle with centered text.
    """
def mjr_maxViewport(con: MjrContext) -> MjrRect:
    """mjr_maxViewport(con: mujoco._render.MjrContext) -> mujoco._render.MjrRect

    Get maximum viewport for active buffer.
    """
def mjr_overlay(font: typing.SupportsInt, gridpos: typing.SupportsInt, viewport: MjrRect, overlay: str, overlay2: str, con: MjrContext) -> None:
    """mjr_overlay(font: typing.SupportsInt, gridpos: typing.SupportsInt, viewport: mujoco._render.MjrRect, overlay: str, overlay2: str, con: mujoco._render.MjrContext) -> None

    Draw text overlay; font is mjtFont; gridpos is mjtGridPos.
    """
def mjr_readPixels(rgb: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8] | None, depth: typing.Annotated[numpy.typing.ArrayLike, numpy.float32] | None, viewport: MjrRect, con: MjrContext) -> None:
    """mjr_readPixels(rgb: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8] | None, depth: typing.Annotated[numpy.typing.ArrayLike, numpy.float32] | None, viewport: mujoco._render.MjrRect, con: mujoco._render.MjrContext) -> None

    Read pixels from current OpenGL framebuffer to client buffer. Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
    """
def mjr_rectangle(viewport: MjrRect, r: typing.SupportsFloat, g: typing.SupportsFloat, b: typing.SupportsFloat, a: typing.SupportsFloat) -> None:
    """mjr_rectangle(viewport: mujoco._render.MjrRect, r: typing.SupportsFloat, g: typing.SupportsFloat, b: typing.SupportsFloat, a: typing.SupportsFloat) -> None

    Draw rectangle.
    """
def mjr_render(viewport: MjrRect, scn: mujoco._structs.MjvScene, con: MjrContext) -> None:
    """mjr_render(viewport: mujoco._render.MjrRect, scn: mujoco._structs.MjvScene, con: mujoco._render.MjrContext) -> None

    Render 3D scene.
    """
def mjr_resizeOffscreen(width: typing.SupportsInt, height: typing.SupportsInt, con: MjrContext) -> None:
    """mjr_resizeOffscreen(width: typing.SupportsInt, height: typing.SupportsInt, con: mujoco._render.MjrContext) -> None

    Resize offscreen buffers.
    """
def mjr_restoreBuffer(con: MjrContext) -> None:
    """mjr_restoreBuffer(con: mujoco._render.MjrContext) -> None

    Make con->currentBuffer current again.
    """
def mjr_setAux(index: typing.SupportsInt, con: MjrContext) -> None:
    """mjr_setAux(index: typing.SupportsInt, con: mujoco._render.MjrContext) -> None

    Set Aux buffer for custom OpenGL rendering (call restoreBuffer when done).
    """
def mjr_setBuffer(framebuffer: typing.SupportsInt, con: MjrContext) -> None:
    """mjr_setBuffer(framebuffer: typing.SupportsInt, con: mujoco._render.MjrContext) -> None

    Set OpenGL framebuffer for rendering: mjFB_WINDOW or mjFB_OFFSCREEN. If only one buffer is available, set that buffer and ignore framebuffer argument.
    """
def mjr_text(font: typing.SupportsInt, txt: str, con: MjrContext, x: typing.SupportsFloat, y: typing.SupportsFloat, r: typing.SupportsFloat, g: typing.SupportsFloat, b: typing.SupportsFloat) -> None:
    """mjr_text(font: typing.SupportsInt, txt: str, con: mujoco._render.MjrContext, x: typing.SupportsFloat, y: typing.SupportsFloat, r: typing.SupportsFloat, g: typing.SupportsFloat, b: typing.SupportsFloat) -> None

    Draw text at (x,y) in relative coordinates; font is mjtFont.
    """
def mjr_uploadHField(m: mujoco._structs.MjModel, con: MjrContext, hfieldid: typing.SupportsInt) -> None:
    """mjr_uploadHField(m: mujoco._structs.MjModel, con: mujoco._render.MjrContext, hfieldid: typing.SupportsInt) -> None

    Upload height field to GPU, overwriting previous upload if any.
    """
def mjr_uploadMesh(m: mujoco._structs.MjModel, con: MjrContext, meshid: typing.SupportsInt) -> None:
    """mjr_uploadMesh(m: mujoco._structs.MjModel, con: mujoco._render.MjrContext, meshid: typing.SupportsInt) -> None

    Upload mesh to GPU, overwriting previous upload if any.
    """
def mjr_uploadTexture(m: mujoco._structs.MjModel, con: MjrContext, texid: typing.SupportsInt) -> None:
    """mjr_uploadTexture(m: mujoco._structs.MjModel, con: mujoco._render.MjrContext, texid: typing.SupportsInt) -> None

    Upload texture to GPU, overwriting previous upload if any.
    """
