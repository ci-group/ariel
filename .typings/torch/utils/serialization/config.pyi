from torch.serialization import LoadEndianness as _LoadEndianess

class load:
    mmap: bool
    endianness: _LoadEndianess | None
    mmap_flags: int | None
    calculate_storage_offsets: bool

class save:
    compute_crc32: bool
    use_pinned_memory_for_d2h: bool
    storage_alignment: int
