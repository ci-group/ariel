import enum

__all__ = ['QuantType']

class QuantType(enum.IntEnum):
    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    WEIGHT_ONLY = 3
