from _typeshed import Incomplete

__all__ = ['Decoder', 'ImageHandler', 'MatHandler', 'audiohandler', 'basichandlers', 'extension_extract_fn', 'handle_extension', 'imagehandler', 'mathandler', 'videohandler']

def basichandlers(extension: str, data):
    """Transforms raw data (byte stream) into python objects.

    Looks at the extension and loads the data into a python object supporting
    the corresponding extension.

    Args:
        extension (str): The file extension
        data (byte stream): Data to load into a python object.

    Returns:
        object: The data loaded into a corresponding python object
            supporting the extension.

    Example:
        >>> import pickle
        >>> data = pickle.dumps('some data')
        >>> new_data = basichandlers('pickle', data)
        >>> new_data
        some data

    The transformation of data for extensions are:
        - txt, text, transcript: utf-8 decoded data of str format
        - cls, cls2, class, count, index, inx, id: int
        - json, jsn: json loaded data
        - pickle, pyd: pickle loaded data
        - pt: torch loaded data
    """
def handle_extension(extensions, f):
    '''
    Return a decoder handler function for the list of extensions.

    Extensions can be a space separated list of extensions.
    Extensions can contain dots, in which case the corresponding number
    of extension components must be present in the key given to f.
    Comparisons are case insensitive.
    Examples:
    handle_extension("jpg jpeg", my_decode_jpg)  # invoked for any file.jpg
    handle_extension("seg.jpg", special_case_jpg)  # invoked only for file.seg.jpg
    '''

class ImageHandler:
    """
    Decode image data using the given `imagespec`.

    The `imagespec` specifies whether the image is decoded
    to numpy/torch/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """
    imagespec: Incomplete
    def __init__(self, imagespec) -> None: ...
    def __call__(self, extension, data): ...

def imagehandler(imagespec): ...
def videohandler(extension, data): ...
def audiohandler(extension, data): ...

class MatHandler:
    sio: Incomplete
    loadmat_kwargs: Incomplete
    def __init__(self, **loadmat_kwargs) -> None: ...
    def __call__(self, extension, data): ...

def mathandler(**loadmat_kwargs): ...
def extension_extract_fn(pathname): ...

class Decoder:
    """
    Decode key/data sets using a list of handlers.

    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """
    handlers: Incomplete
    key_fn: Incomplete
    def __init__(self, *handler, key_fn=...) -> None: ...
    def add_handler(self, *handler) -> None: ...
    @staticmethod
    def _is_stream_handle(data): ...
    def decode1(self, key, data): ...
    def decode(self, data): ...
    def __call__(self, data): ...
