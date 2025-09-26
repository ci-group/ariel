from _typeshed import Incomplete

__all__ = ['LinearBlockSparsePattern']

class LinearBlockSparsePattern:
    rlock: Incomplete
    row_block_size: int
    col_block_size: int
    prev_row_block_size: int
    prev_col_block_size: int
    def __init__(self, row_block_size: int = 1, col_block_size: int = 4) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, backtrace: object | None) -> None: ...
    @staticmethod
    def block_size() -> tuple[int, int]: ...
