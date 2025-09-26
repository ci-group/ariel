__all__ = ['AmbiguityWarning', 'supercedes', 'consistent', 'ambiguous', 'ambiguities', 'super_signature', 'edge', 'ordering']

class AmbiguityWarning(Warning): ...

def supercedes(a, b):
    """A is consistent and strictly more specific than B"""
def consistent(a, b):
    """It is possible for an argument list to satisfy both A and B"""
def ambiguous(a, b):
    """A is consistent with B but neither is strictly more specific"""
def ambiguities(signatures):
    """All signature pairs such that A is ambiguous with B"""
def super_signature(signatures):
    """A signature that would break ambiguities"""
def edge(a, b, tie_breaker=...):
    """A should be checked before B
    Tie broken by tie_breaker, defaults to ``hash``
    """
def ordering(signatures):
    """A sane ordering of signatures to check, first to last
    Topological sort of edges as given by ``edge`` and ``supercedes``
    """
