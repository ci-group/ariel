import logging

_log_handlers: dict[str, logging.Handler]

def get_logging_handler(destination: str = 'null') -> logging.Handler: ...
