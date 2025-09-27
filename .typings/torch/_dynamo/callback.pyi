import enum
import threading
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

class CallbackTrigger(enum.Enum):
    DYNAMO = 1
    LAZY_BACKWARD = 2
    TRITON_AUTOTUNING = 3
    CUDAGRAPH_RECORDING = 4

@dataclass
class CallbackArgs:
    callback_trigger: CallbackTrigger
    compile_id: str

@dataclass
class CompilationCallbackHandler:
    start_callbacks: list[Callable[[CallbackArgs], None]] = field(default_factory=list)
    end_callbacks: list[Callable[[CallbackArgs], None]] = field(default_factory=list)
    __pending_callbacks_counter: int = field(default=0, init=False, repr=False)
    __pending_callbacks_counter_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    def register_start_callback(self, callback: Callable[[CallbackArgs], None]) -> Callable[[CallbackArgs], None]:
        """
        Register a callback function to be called when the compilation starts.

        Args:
        - callback (Callable): The callback function to register.
        """
    def register_end_callback(self, callback: Callable[[CallbackArgs], None]) -> Callable[[CallbackArgs], None]:
        """
        Register a callback function to be called when the compilation ends.

        Args:
        - callback (Callable): The callback function to register.
        """
    def remove_start_callback(self, callback: Callable[[CallbackArgs], None]) -> None:
        """
        Remove a registered start callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
    def remove_end_callback(self, callback: Callable[[CallbackArgs], None]) -> None:
        """
        Remove a registered end callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
    def run_start_callbacks(self, args: CallbackArgs) -> None:
        """
        Execute all registered start callbacks.
        """
    def run_end_callbacks(self, args: CallbackArgs) -> None:
        """
        Execute all registered end callbacks.
        """
    @contextmanager
    def install_callbacks(self, trigger: CallbackTrigger, compile_id: str) -> Generator[None, Any, Any]:
        """
        Context manager to install the callbacks and run them when the context is exited.
        """
    def clear(self) -> None:
        """
        Clear all registered callbacks.
        """

callback_handler: Incomplete

def on_compile_start(callback: Callable[[CallbackArgs], None]) -> Callable[[CallbackArgs], None]:
    """
    Decorator to register a callback function for the start of the compilation.
    """
def on_compile_end(callback: Callable[[CallbackArgs], None]) -> Callable[[CallbackArgs], None]:
    """
    Decorator to register a callback function for the end of the compilation.
    """
