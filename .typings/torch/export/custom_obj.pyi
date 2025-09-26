from dataclasses import dataclass

__all__ = ['ScriptObjectMeta']

@dataclass
class ScriptObjectMeta:
    """
    Metadata which is stored on nodes representing ScriptObjects.
    """
    constant_name: str
    class_fqn: str
