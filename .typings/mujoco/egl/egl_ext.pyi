from OpenGL.EGL import *
from _typeshed import Incomplete

EGL_CUDA_DEVICE_NV: int
PFNEGLQUERYDEVICESEXTPROC: Incomplete
EGL_PLATFORM_DEVICE_EXT: int
PFNEGLGETPLATFORMDISPLAYEXTPROC: Incomplete
eglGetPlatformDisplayEXT: Incomplete
PFNEGLQUERYDEVICEATTRIBEXTPROC: Incomplete

def eglQueryDevicesEXT(max_devices: int = 10): ...
def eglQueryDeviceAttribEXT(device, attribute): ...
