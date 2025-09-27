import socket

__all__ = ['create_c10d_store', 'get_free_port', 'get_socket_with_port']

def create_c10d_store(is_server: bool, server_addr: str, server_port: int = -1, world_size: int = 1, timeout: float = ..., wait_for_workers: bool = True, retries: int = 3, use_libuv: bool | None = None): ...
def get_free_port():
    '''
    Returns an unused port on localhost.

    This function finds an unused port on localhost by opening to socket to bind
    to a port and then closing it.

    Returns:
        int: an unused port on localhost

    Example:
        >>> # xdoctest: +SKIP("Nondeterministic")
        >>> get_free_port()
        63976

    .. note::
        The port returned by :func:`get_free_port` is not reserved and may be
        taken by another process after this function returns.
    '''
def get_socket_with_port() -> socket.socket:
    '''
    Returns a free port on localhost that is "reserved" by binding a temporary
    socket on it. Close the socket before passing the port to the entity
    that requires it. Usage example

    ::

    sock = _get_socket_with_port()
    with closing(sock):
        port = sock.getsockname()[1]
        sock.close()
        # there is still a race-condition that some other process
        # may grab this port before func() runs
        func(port)
    '''
