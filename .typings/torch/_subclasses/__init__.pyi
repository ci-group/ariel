from torch._subclasses.fake_tensor import DynamicOutputShapeException as DynamicOutputShapeException, FakeTensor as FakeTensor, FakeTensorMode as FakeTensorMode, UnsupportedFakeTensorException as UnsupportedFakeTensorException
from torch._subclasses.fake_utils import CrossRefFakeMode as CrossRefFakeMode

__all__ = ['FakeTensor', 'FakeTensorMode', 'UnsupportedFakeTensorException', 'DynamicOutputShapeException', 'CrossRefFakeMode']
