from torch.fx.experimental.proxy_tensor import PythonKeyTracer as PythonKeyTracer, decompose, dispatch_trace as dispatch_trace, make_fx as make_fx

__all__ = ['make_fx', 'dispatch_trace', 'PythonKeyTracer', 'pythonkey_decompose']

pythonkey_decompose = decompose
