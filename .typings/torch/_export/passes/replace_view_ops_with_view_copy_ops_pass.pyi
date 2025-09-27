from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse

__all__ = ['ReplaceViewOpsWithViewCopyOpsPass']

class ReplaceViewOpsWithViewCopyOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Our backend expects pure functional operators. For efficiency
    purposes, we keep view ops around while functionalizing the exported
    program. This pass replaces view ops with view copy ops for backends that
    need AOT memory planning.
    """
    def call_operator(self, op, args, kwargs, meta): ...
