import contextlib
import types
from _typeshed import Incomplete
from dataclasses import dataclass
from torch._utils_internal import justknobs_check as justknobs_check
from types import ModuleType
from typing import Any, Callable, Generic, NoReturn, TypeVar

CONFIG_TYPES: Incomplete
T = TypeVar('T', bound=int | float | bool | None | str | list | set | tuple | dict)
_UNSET_SENTINEL: Incomplete

@dataclass
class _Config(Generic[T]):
    '''Represents a config with richer behaviour than just a default value.
    ::
        i.e.
        foo = Config(justknob="//foo:bar", default=False)
        install_config_module(...)

    This configs must be installed with install_config_module to be used

    Precedence Order:
        alias: If set, the directly use the value of the alias.
        env_name_force: If set, this environment variable has precedence over
            everything after this.
            If multiple env variables are given, the precendence order is from
            left to right.
        user_override: If a user sets a value (i.e. foo.bar=True), that
            has precedence over everything after this.
        env_name_default: If set, this environment variable will override everything
            after this.
            If multiple env variables are given, the precendence order is from
            left to right.
        justknob: If this pytorch installation supports justknobs, that will
            override defaults, but will not override the user_override precendence.
        default: This value is the lowest precendance, and will be used if nothing is
            set.

    Environment Variables:
        These are interpreted to be either "0" or "1" to represent true and false.

    Arguments:
        justknob: the name of the feature / JK. In OSS this is unused.
        default: is the value to default this knob to in OSS.
        alias: The alias config to read instead.
        env_name_force: The environment variable, or list of, to read that is a FORCE
            environment variable. I.e. it overrides everything except for alias.
        env_name_default: The environment variable, or list of, to read that changes the
            default behaviour. I.e. user overrides take preference.
    '''
    default: T | object
    justknob: str | None = ...
    env_name_default: list[str] | None = ...
    env_name_force: list[str] | None = ...
    alias: str | None = ...
    value_type = ...
    def __init__(self, default: T | object = ..., justknob: str | None = None, env_name_default: str | list[str] | None = None, env_name_force: str | list[str] | None = None, value_type: type | None = None, alias: str | None = None) -> None: ...
    @staticmethod
    def string_or_list_of_string_to_list(val: str | list[str] | None) -> list[str] | None: ...

def Config(default: T | object = ..., justknob: str | None = None, env_name_default: str | list[str] | None = None, env_name_force: str | list[str] | None = None, value_type: type | None = None, alias: str | None = None) -> T: ...
def _read_env_variable(name: str) -> bool | str | None: ...
def install_config_module(module: ModuleType) -> None:
    """
    Converts a module-level config into a `ConfigModule()`.

    See _config_typing.pyi for instructions on how to get the converted module to typecheck.
    """

COMPILE_IGNORED_MARKER: str

def get_assignments_with_compile_ignored_comments(module: ModuleType) -> set[str]: ...

@dataclass
class _ConfigEntry:
    default: Any
    value_type: type
    user_override: Any = ...
    justknob: str | None = ...
    env_value_force: Any = ...
    env_value_default: Any = ...
    hide: bool = ...
    alias: str | None = ...
    def __init__(self, config: _Config) -> None: ...

class ConfigModule(ModuleType):
    _config: dict[str, _ConfigEntry]
    _bypass_keys: set[str]
    _compile_ignored_keys: set[str]
    _is_dirty: bool
    _hash_digest: bytes | None
    def __init__(self) -> None: ...
    def __setattr__(self, name: str, value: object) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __delattr__(self, name: str) -> None: ...
    def _get_alias_module_and_name(self, entry: _ConfigEntry) -> tuple[ModuleType, str] | None: ...
    def _get_alias_val(self, entry: _ConfigEntry) -> Any: ...
    def _set_alias_val(self, entry: _ConfigEntry, val: Any) -> None: ...
    def _is_default(self, name: str) -> bool:
        """
        Returns true if the config is at its default value.
        configs overridden by the env are not considered default.
        """
    def _get_dict(self, ignored_keys: list[str] | None = None, ignored_prefixes: list[str] | None = None, skip_default: bool = False) -> dict[str, Any]:
        """Export a dictionary of current configuration keys and values.

        This function is design to provide a single point which handles
        accessing config options and exporting them into a dictionary.
        This is used by a number of different user facing export methods
        which all have slightly different semantics re: how and what to
        skip.
        If a config is aliased, it skips this config.

        Arguments:
            ignored_keys are keys that should not be exported.
            ignored_prefixes are prefixes that if a key matches should
                not be exported
            skip_default does two things. One if a key has not been modified
                it skips it.
        """
    def get_type(self, config_name: str) -> type: ...
    def save_config(self) -> bytes:
        """Convert config to a pickled blob"""
    def save_config_portable(self, *, ignore_private_configs: bool = True) -> dict[str, Any]:
        """Convert config to portable format"""
    def codegen_config(self) -> str:
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """
    def get_hash(self) -> bytes:
        """Hashes the configs that are not compile_ignored"""
    def to_dict(self) -> dict[str, Any]: ...
    def shallow_copy_dict(self) -> dict[str, Any]: ...
    def load_config(self, maybe_pickled_config: bytes | dict[str, Any]) -> None:
        """Restore from a prior call to save_config() or shallow_copy_dict()"""
    def get_config_copy(self) -> dict[str, Any]: ...
    changes: Incomplete
    def patch(self, arg1: str | dict[str, Any] | None = None, arg2: Any = None, **kwargs: dict[str, Any]) -> ContextDecorator:
        '''
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2)
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        '''
    def _make_closure_patcher(self, **changes: dict[str, Any]) -> Any:
        """
        A lower-overhead version of patch() for things on the critical path.

        Usage:

            # do this off the critical path
            change_fn = config.make_closure_patcher(foo=True)

            ...

            revert = change_fn()
            try:
              ...
            finally:
                revert()

        """

class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> NoReturn: ...
    def __call__(self, func: Callable[[Any], Any]) -> Any: ...

class SubConfigProxy:
    '''
    Shim to redirect to main config.
    `config.triton.cudagraphs` maps to _config["triton.cudagraphs"]
    '''
    def __init__(self, config: object, prefix: str) -> None: ...
    def __setattr__(self, name: str, value: object) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __delattr__(self, name: str) -> None: ...

def patch_object(obj: object, name: str, value: object) -> object:
    """
    Workaround `mock.patch.object` issue with ConfigModule
    """
def get_tristate_env(name: str, default: Any = None) -> bool | None: ...
