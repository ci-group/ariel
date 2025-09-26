import dataclasses
import torch
import typing
import typing_extensions
from torch.export.exported_program import _decompose_exported_program as _decompose_exported_program

def _copy_graph_module_and_signature(ep: torch.fx.GraphModule) -> tuple[torch.fx.GraphModule, torch.export.graph_signature.ExportGraphSignature]: ...
def _remove_detach_pass(gm: torch.fx.GraphModule, sig: torch.export.graph_signature.ExportGraphSignature) -> None: ...
def _export_forward_backward(ep: torch.export.ExportedProgram, joint_loss_index: int = 0) -> torch.export.ExportedProgram:
    """
    WARNING: This API is highly unstable and will be subject to change in the future.
    """
@typing.no_type_check
def _sticky_export(forward_func, dynamic_shapes_callback=None):
    """
    Lazily export the model on first forward call.
    Usage:
        model.forward = _sticky_export(model.forward, dynamic_shapes_callback=callback)
    """

@dataclasses.dataclass
class _ExportMethod:
    overloads: dict[str, torch.export.ExportedProgram]
    fallbacks: list[torch.export.ExportedProgram]
_InputT = typing_extensions.ParamSpec('_InputT')
_RetT = typing.TypeVar('_RetT')

class _ExportPackage:
    '''
    An export package is a collection of torch.export()-ed PyTorch models consisting of
    a list of exported methods and their corresponding overloads. ExportPackage is introduced
    on top of torch.export() to support the following use cases:
        - Exporting a model with multiple methods if a model has multiple independent parts.
        - Exporting a function with multiple overloads based on tensor shapes or other metadata.

    ExportPackage is designed to contain multiple methods (associated with method names) and for
    each method, it can have multiple overloads (associated with overload names).

    Here is an example of the data structure for an ExportPackage:
    ```
    ExportPackage(
        methods={
            "decoder": ExportMethod(
                overloads={
                    "prefill": ExportedProgram(...),
                    "decode": ExportedProgram(...),
                },
                fallbacks=[],
            ),
            "encoder": ExportMethod(overloads={}, fallbacks=[ExportedProgram(...)]),
        },
    )
    ```

    To export a model into an ExportPackage, users can use the exporter API provided by ExportPackage.
    Exporter is a decorator that takes a callable and returns a wrapper. The wrapper will export the
    function into an ExportPackage, when it\'s invoked with some sample inputs (similar to how
    torch.compile() works). For more details, please refer to the document on .exporter() method.

    This design allows users to decouple the exported callables from the actual sample inputs which can
    be helpful for use cases where the exported callable is hidden behind helper functions or when sample
    inpusts are hard to get.

    NOTE: This is an experimental API and anything can be changed in the future.

    Example usage:
    ```
        def fn(x):
            return x + 1

        def main(f, x):
            x += 1
            ret = f(x)
            return ret + 1

        package = ExportPackage()
        main(package.exporter(fn), torch.randn(3, 2))
    ```

    '''
    methods: dict[str, _ExportMethod]
    def __init__(self) -> None: ...
    def _exporter(self, method: str, fn: typing.Callable[_InputT, _RetT], *, fallback: str = 'once') -> typing.Callable[_InputT, _RetT]:
        '''
        A function/module decorator that sets up a callable to be exported later invoked.
        By default the exporter will only trigger torch.export for once and error on
        later invocations. To customize this behavior, users have the following two options:
          1. Call .define_overload() method on the returned wrapper to define an overload.
          2. Adjust the fallback policy using `fallback` argument.

        An "overload" is a named branch for an ExportMethod with a user defined precondition,
        typically based on input tensor shapes. It\'s up to a downstream backend implementation
        of ExportMethod to respect the precondition later in inference.

        define_overload() takes arguments like the following:
          - A name, for indexing purposes in a backend.
          - A callable (spec) that:
            - Has the same model input signature as the original model code.
            - Returns an optional dynamic shape spec.

        Exporter will only export an overload when the spec callable successfully returns
        a result without rasing AssertionError.

        For example:
        ```
        package = ExportPackage()


        def prefill(x, xa, kv_cache):
            assert x.shape[1] == 3
            assert kv_cache == {}


        def decode(x, xa, kv_cache):
            assert x.shape[1] > 1
            assert len(kv_cache) > 0
            return {...}  # dynamic shape specs here


        exporter = (
            package.exporter(decoder)
            .define_overload("prefill", prefill)
            .define_overload("decode", decode)
        )
        ```

        A "fallback" is exported when no overload precondition matches a given set of sample
        inputs. Overloads should
        Fallbacks don\'t have names and are ordered in a list. It\'s up to a backend to decide
        which fallback is used amony multiple ones.

        A reference backend implementation of ExportMethod may look like the following:
        ```
        def execute(method: ExportMethod, *args, **kwargs):
            for overload in method.overloads:
                if match_precondition(overload, *args, **kwargs):
                    return execute_overload(overload, *args, **kwargs)
            for fallback in method.fallbacks:
                if match_precondition(fallback, *args, **kwargs):
                    return execute_fallback(fallback, *args, **kwargs)
        ```

        Args:
            method(str): The method name for an exported part of PyTorch model. This
                         will be saved together with the exported/compiled artifacts
                         in any serialization format and can be used as the key to
                         index ExportPackage methods later.
            fn(callable): A PyTorch function/module to be exported.
            fallback(str): The fallback policy to decide when to call torch.export
              - "once" is the default policy. Under this policy a PyTorch program is assumed
                to be only called once later and an error will be raised for subsequent
                runs.
              - "error" means the ExportMethod will never have any fallbacks, meaning
                users should define all the possible overloads ahead of time.

        '''
