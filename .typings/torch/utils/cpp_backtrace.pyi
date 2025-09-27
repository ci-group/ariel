from torch._C import _get_cpp_backtrace as _get_cpp_backtrace

def get_cpp_backtrace(frames_to_skip: int = 0, maximum_number_of_frames: int = 64) -> str:
    """
    Return a string containing the C++ stack trace of the current thread.

    Args:
        frames_to_skip (int): the number of frames to skip from the top of the stack
        maximum_number_of_frames (int): the maximum number of frames to return
    """
