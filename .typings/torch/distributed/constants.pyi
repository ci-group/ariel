from datetime import timedelta

__all__ = ['default_pg_timeout', 'default_pg_nccl_timeout']

default_pg_timeout: timedelta
default_pg_nccl_timeout: timedelta | None
