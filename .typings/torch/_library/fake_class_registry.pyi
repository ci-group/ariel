import torch
from _typeshed import Incomplete
from torch._library.utils import parse_namespace as parse_namespace
from torch.utils._python_dispatch import _disable_current_modes as _disable_current_modes
from typing import Any, Protocol

log: Incomplete

class FakeScriptObject:
    wrapped_obj: Incomplete
    script_class_name: Incomplete
    real_obj: Incomplete
    def __init__(self, wrapped_obj: Any, script_class_name: str, x: torch.ScriptObject) -> None: ...

class FakeScriptMethod:
    self_fake_obj: Incomplete
    method_name: Incomplete
    schema: Incomplete
    def __init__(self, self_fake_obj: FakeScriptObject, method_name: str, schema: torch.FunctionSchema | None) -> None: ...
    def __call__(self, *args, **kwargs): ...

class HasStaticMethodFromReal(Protocol):
    @classmethod
    def from_real(cls, real_obj: torch.ScriptObject): ...

class FakeClassRegistry:
    _registered_class: dict[str, Any]
    def __init__(self) -> None: ...
    def has_impl(self, full_qualname: str) -> bool: ...
    def get_impl(self, full_qualname: str) -> Any: ...
    def register(self, full_qualname: str, fake_class=None) -> None: ...
    def deregister(self, full_qualname: str) -> Any: ...
    def clear(self) -> None: ...
    def _check_registered(self, full_qualname: str) -> None: ...

global_fake_class_registry: Incomplete

def _check_valid_flat_script_obj(flat_x) -> None: ...
def tracing_with_real(x: torch.ScriptObject) -> bool: ...
def maybe_to_fake_obj(fake_mode, x: torch.ScriptObject) -> FakeScriptObject | torch.ScriptObject: ...
def register_fake_class(qualname, fake_class: HasStaticMethodFromReal | None = None):
    '''Register a fake implementation for this class.

    It\'s in the same spirit of registering a fake implementation for
    an operator but with the difference that it
    associates a fake class with the original torch bind class (registered
    with torch::class_). In this way, torch.compile can handle them properly
    in components such as Dynamo and AOTAutograd.

    This API may be used as a decorator (see example). For the fake class, users
    are required to provide a from_real classmethod that takes a real object and
    returns an instance of the fake class. All tensors in the fake object should also
    be properly fakified with to_fake_tensor() in from_real.


    Examples:
        # For a custom class Foo defined in test_custom_class_registration.cpp:

        TORCH_LIBRARY(_TorchScriptTesting, m) {
          m.class_<TensorQueue>("_TensorQueue")
            .def(torch::init<at::Tensor>())
            .def("push", &TensorQueue::push)
            .def("pop", &TensorQueue::pop)
            .def("top", &TensorQueue::top)
            .def("size", &TensorQueue::size)
            .def("clone_queue", &TensorQueue::clone_queue)
            .def("__obj_flatten__", &TensorQueue::__obj_flatten__)
            .def_pickle(
                // __getstate__
                [](const c10::intrusive_ptr<TensorQueue>& self)
                    -> c10::Dict<std::string, at::Tensor> {
                  return self->serialize();
                },
                // __setstate__
                [](c10::Dict<std::string, at::Tensor> data)
                    -> c10::intrusive_ptr<TensorQueue> {
                  return c10::make_intrusive<TensorQueue>(std::move(data));
                });
            };
        # We could register a fake class FakeTensorQueue in Python as follows:
        import torch

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            def __init__(self, queue):
                self.queue = queue

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                return cls(**dict(ctx))

            def push(self, x):
                self.queue.append(x)

            def pop(self):
                return self.queue.pop(0)

            def size(self):
                return len(self.queue)

    In this example, the original TensorQeue need to addd a __obj_flatten__ method
    to the class TensorQueue and the flattend result is passed into FakeTensorQueue\'s
    __obj_unflatten__ as inputs to create a fake class. This protocol allows pytorch to look
    at the contents of the script object and properly handle them in the subsystems
    like dynamo, aot_aotugrad or more.
    '''
def deregister_fake_class(qualname): ...
def has_fake_class(full_qualname) -> bool: ...
def find_fake_class(full_qualname) -> Any | None: ...
def _full_qual_class_name(qualname: str) -> str: ...
def _is_script_object(obj: Any) -> bool: ...
def _ns_and_class_name(full_qualname: str) -> tuple[str, str]: ...
def _find_fake_class_for_script_object(x: torch.ScriptObject) -> Any: ...

_CONVERT_FROM_REAL_NAME: str

def _fake_obj_from_real(fake_mode, x) -> Any: ...
