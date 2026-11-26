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

        2. __init__(self: mujoco._render.MjrContext, arg0: mujoco._structs.MjModel, arg1: typing.SupportsInt | typing.SupportsIndex) -> None
        """
    @overload
    def __init__(self, arg0: mujoco._structs.MjModel, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._render.MjrContext) -> None

        2. __init__(self: mujoco._render.MjrContext, arg0: mujoco._structs.MjModel, arg1: typing.SupportsInt | typing.SupportsIndex) -> None
        """
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

class MjrRendererInfo:
    def __init__(self) -> None:
        """__init__(self: mujoco._render.MjrRendererInfo) -> None"""
    def __copy__(self) -> MjrRendererInfo:
        """__copy__(self: mujoco._render.MjrRendererInfo) -> mujoco._render.MjrRendererInfo"""
    def __deepcopy__(self, arg0: dict) -> MjrRendererInfo:
        """__deepcopy__(self: mujoco._render.MjrRendererInfo, arg0: dict) -> mujoco._render.MjrRendererInfo"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @property
    def backend(self) -> str:
        """(arg0: mujoco._render.MjrRendererInfo) -> str"""
    @property
    def renderer(self) -> str:
        """(arg0: mujoco._render.MjrRendererInfo) -> str"""

def mjr_addAux(index: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex, samples: typing.SupportsInt | typing.SupportsIndex, con: MjrContext) -> None:
    """mjr_addAux(index: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex, samples: typing.SupportsInt | typing.SupportsIndex, con: mujoco._render.MjrContext) -> None

    Add Aux buffer with given index to context; free previous Aux buffer.
    """
def mjr_blitAux(index: typing.SupportsInt | typing.SupportsIndex, src: mujoco._structs.MjrRect, left: typing.SupportsInt | typing.SupportsIndex, bottom: typing.SupportsInt | typing.SupportsIndex, con: MjrContext) -> None:
    """mjr_blitAux(index: typing.SupportsInt | typing.SupportsIndex, src: mujoco._structs.MjrRect, left: typing.SupportsInt | typing.SupportsIndex, bottom: typing.SupportsInt | typing.SupportsIndex, con: mujoco._render.MjrContext) -> None

    Blit from Aux buffer to con->currentBuffer.
    """
def mjr_blitBuffer(src: mujoco._structs.MjrRect, dst: mujoco._structs.MjrRect, flg_color: typing.SupportsInt | typing.SupportsIndex, flg_depth: typing.SupportsInt | typing.SupportsIndex, con: MjrContext) -> None:
    """mjr_blitBuffer(src: mujoco._structs.MjrRect, dst: mujoco._structs.MjrRect, flg_color: typing.SupportsInt | typing.SupportsIndex, flg_depth: typing.SupportsInt | typing.SupportsIndex, con: mujoco._render.MjrContext) -> None

    Blit from src viewpoint in current framebuffer to dst viewport in other framebuffer. If src, dst have different size and flg_depth==0, color is interpolated with GL_LINEAR.
    """
def mjr_changeFont(fontscale: typing.SupportsInt | typing.SupportsIndex, con: MjrContext) -> None:
    """mjr_changeFont(fontscale: typing.SupportsInt | typing.SupportsIndex, con: mujoco._render.MjrContext) -> None

    Change font of existing context.
    """
def mjr_defaultRendererInfo(info: MjrRendererInfo) -> None:
    """mjr_defaultRendererInfo(info: mujoco._render.MjrRendererInfo) -> None

    Set default mjrRendererInfo.
    """
def mjr_drawPixels(rgb: typing.Annotated[numpy.typing.NDArray[numpy.uint8], '[m, 1]'] | None, depth: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[m, 1]'] | None, viewport: mujoco._structs.MjrRect, con: MjrContext) -> None:
    '''mjr_drawPixels(rgb: typing.Annotated[numpy.typing.NDArray[numpy.uint8], "[m, 1]"] | None, depth: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, 1]"] | None, viewport: mujoco._structs.MjrRect, con: mujoco._render.MjrContext) -> None

    Draw pixels from client buffer to current OpenGL framebuffer. Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
    '''
def mjr_figure(viewport: mujoco._structs.MjrRect, fig: mujoco._structs.MjvFigure, con: MjrContext) -> None:
    """mjr_figure(viewport: mujoco._structs.MjrRect, fig: mujoco._structs.MjvFigure, con: mujoco._render.MjrContext) -> None

    Draw 2D figure.
    """
def mjr_findRect(x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex, nrect: typing.SupportsInt | typing.SupportsIndex, rect: mujoco._structs.MjrRect) -> int:
    """mjr_findRect(x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex, nrect: typing.SupportsInt | typing.SupportsIndex, rect: mujoco._structs.MjrRect) -> int

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
def mjr_getRendererInfo(info: MjrRendererInfo) -> None:
    """mjr_getRendererInfo(info: mujoco._render.MjrRendererInfo) -> None

    Get active renderer information.
    """
def mjr_label(viewport: mujoco._structs.MjrRect, font: typing.SupportsInt | typing.SupportsIndex, txt: str, r: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, rt: typing.SupportsFloat | typing.SupportsIndex, gt: typing.SupportsFloat | typing.SupportsIndex, bt: typing.SupportsFloat | typing.SupportsIndex, con: MjrContext) -> None:
    """mjr_label(viewport: mujoco._structs.MjrRect, font: typing.SupportsInt | typing.SupportsIndex, txt: str, r: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, rt: typing.SupportsFloat | typing.SupportsIndex, gt: typing.SupportsFloat | typing.SupportsIndex, bt: typing.SupportsFloat | typing.SupportsIndex, con: mujoco._render.MjrContext) -> None

    Draw rectangle with centered text.
    """
def mjr_maxViewport(con: MjrContext) -> mujoco._structs.MjrRect:
    """mjr_maxViewport(con: mujoco._render.MjrContext) -> mujoco._structs.MjrRect

    Get maximum viewport for active buffer.
    """
def mjr_overlay(font: typing.SupportsInt | typing.SupportsIndex, gridpos: typing.SupportsInt | typing.SupportsIndex, viewport: mujoco._structs.MjrRect, overlay: str, overlay2: str, con: MjrContext) -> None:
    """mjr_overlay(font: typing.SupportsInt | typing.SupportsIndex, gridpos: typing.SupportsInt | typing.SupportsIndex, viewport: mujoco._structs.MjrRect, overlay: str, overlay2: str, con: mujoco._render.MjrContext) -> None

    Draw text overlay; font is mjtFont; gridpos is mjtGridPos.
    """
def mjr_readPixels(rgb: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8] | None, depth: typing.Annotated[numpy.typing.ArrayLike, numpy.float32] | None, viewport: mujoco._structs.MjrRect, con: MjrContext) -> None:
    """mjr_readPixels(rgb: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8] | None, depth: typing.Annotated[numpy.typing.ArrayLike, numpy.float32] | None, viewport: mujoco._structs.MjrRect, con: mujoco._render.MjrContext) -> None

    Read pixels from current OpenGL framebuffer to client buffer. Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
    """
def mjr_rectangle(viewport: mujoco._structs.MjrRect, r: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """mjr_rectangle(viewport: mujoco._structs.MjrRect, r: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex) -> None

    Draw rectangle.
    """
def mjr_render(viewport: mujoco._structs.MjrRect, scn: mujoco._structs.MjvScene, con: MjrContext) -> None:
    """mjr_render(viewport: mujoco._structs.MjrRect, scn: mujoco._structs.MjvScene, con: mujoco._render.MjrContext) -> None

    Render 3D scene.
    """
def mjr_resizeOffscreen(width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex, con: MjrContext) -> None:
    """mjr_resizeOffscreen(width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex, con: mujoco._render.MjrContext) -> None

    Resize offscreen buffers.
    """
def mjr_restoreBuffer(con: MjrContext) -> None:
    """mjr_restoreBuffer(con: mujoco._render.MjrContext) -> None

    Make con->currentBuffer current again.
    """
def mjr_setAux(index: typing.SupportsInt | typing.SupportsIndex, con: MjrContext) -> None:
    """mjr_setAux(index: typing.SupportsInt | typing.SupportsIndex, con: mujoco._render.MjrContext) -> None

    Set Aux buffer for custom OpenGL rendering (call restoreBuffer when done).
    """
def mjr_setBuffer(framebuffer: typing.SupportsInt | typing.SupportsIndex, con: MjrContext) -> None:
    """mjr_setBuffer(framebuffer: typing.SupportsInt | typing.SupportsIndex, con: mujoco._render.MjrContext) -> None

    Set OpenGL framebuffer for rendering: mjFB_WINDOW or mjFB_OFFSCREEN. If only one buffer is available, set that buffer and ignore framebuffer argument.
    """
def mjr_text(font: typing.SupportsInt | typing.SupportsIndex, txt: str, con: MjrContext, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """mjr_text(font: typing.SupportsInt | typing.SupportsIndex, txt: str, con: mujoco._render.MjrContext, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> None

    Draw text at (x,y) in relative coordinates; font is mjtFont.
    """
def mjr_uploadHField(m: mujoco._structs.MjModel, con: MjrContext, hfieldid: typing.SupportsInt | typing.SupportsIndex) -> None:
    """mjr_uploadHField(m: mujoco._structs.MjModel, con: mujoco._render.MjrContext, hfieldid: typing.SupportsInt | typing.SupportsIndex) -> None

    Upload height field to GPU, overwriting previous upload if any.
    """
def mjr_uploadMesh(m: mujoco._structs.MjModel, con: MjrContext, meshid: typing.SupportsInt | typing.SupportsIndex) -> None:
    """mjr_uploadMesh(m: mujoco._structs.MjModel, con: mujoco._render.MjrContext, meshid: typing.SupportsInt | typing.SupportsIndex) -> None

    Upload mesh to GPU, overwriting previous upload if any.
    """
def mjr_uploadTexture(m: mujoco._structs.MjModel, con: MjrContext, texid: typing.SupportsInt | typing.SupportsIndex) -> None:
    """mjr_uploadTexture(m: mujoco._structs.MjModel, con: mujoco._render.MjrContext, texid: typing.SupportsInt | typing.SupportsIndex) -> None

    Upload texture to GPU, overwriting previous upload if any. A by-reference texture (tex_adr == -1) is decoded with mj_loadTexture.
    """
