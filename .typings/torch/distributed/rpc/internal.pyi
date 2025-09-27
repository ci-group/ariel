import pickle
from _typeshed import Incomplete
from enum import Enum
from typing import NamedTuple

__all__ = ['RPCExecMode', 'serialize', 'deserialize', 'PythonUDF', 'RemoteException']

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler

class RPCExecMode(Enum):
    SYNC = 'sync'
    ASYNC = 'async'
    ASYNC_JIT = 'async_jit'
    REMOTE = 'remote'

class _InternalRPCPickler:
    '''
    This class provides serialize() and deserialize() interfaces to serialize
    data to be "binary string + tensor table" format
    So for RPC python UDF function and args, non tensor data will be serialized
    into regular binary string, tensor data will be put into thread local tensor
    tables, this serialization format is consistent with builtin operator and args
    using JIT pickler. This format will make tensor handling in C++ much easier,
    e.g. attach tensor to distributed autograd graph in C++
    '''
    _dispatch_table: Incomplete
    _class_reducer_dict: Incomplete
    def __init__(self) -> None: ...
    def _register_reducer(self, obj_class, reducer) -> None: ...
    @classmethod
    def _tensor_receiver(cls, tensor_index): ...
    def _tensor_reducer(self, tensor): ...
    @classmethod
    def _py_rref_receiver(cls, rref_fork_data): ...
    def _py_rref_reducer(self, py_rref): ...
    def _rref_reducer(self, rref): ...
    @classmethod
    def _script_module_receiver(cls, script_module_serialized):
        """
        Given a serialized representation of a ScriptModule created with torch.jit.save,
        loads and returns the ScriptModule.
        """
    def _script_module_reducer(self, script_module):
        """
        Serializes a ScriptModule.
        """
    def serialize(self, obj):
        """
        Serialize non tensor data into binary string, tensor data into
        tensor table
        """
    def deserialize(self, binary_data, tensor_table):
        """
        Deserialize binary string + tensor table to original obj
        """

def serialize(obj): ...
def deserialize(binary_data, tensor_table): ...

class PythonUDF(NamedTuple):
    func: Incomplete
    args: Incomplete
    kwargs: Incomplete

class RemoteException(NamedTuple):
    msg: Incomplete
    exception_type: Incomplete

# Names in __all__ with no definition:
#   PythonUDF
#   RemoteException
