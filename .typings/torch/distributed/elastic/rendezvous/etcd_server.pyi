import subprocess
from _typeshed import Incomplete
from typing import TextIO

logger: Incomplete

def find_free_port():
    '''
    Find a free port and binds a temporary socket to it so that the port can be "reserved" until used.

    .. note:: the returned socket must be closed before using the port,
              otherwise a ``address already in use`` error will happen.
              The socket should be held and closed as close to the
              consumer of the port as possible since otherwise, there
              is a greater chance of race-condition where a different
              process may see the port as being free and take it.

    Returns: a socket binded to the reserved free port

    Usage::

    sock = find_free_port()
    port = sock.getsockname()[1]
    sock.close()
    use_port(port)
    '''
def stop_etcd(subprocess, data_dir: str | None = None): ...

class EtcdServer:
    '''
    .. note:: tested on etcd server v3.4.3.

    Starts and stops a local standalone etcd server on a random free
    port. Useful for single node, multi-worker launches or testing,
    where a sidecar etcd server is more convenient than having to
    separately setup an etcd server.

    This class registers a termination handler to shutdown the etcd
    subprocess on exit. This termination handler is NOT a substitute for
    calling the ``stop()`` method.

    The following fallback mechanism is used to find the etcd binary:

    1. Uses env var TORCHELASTIC_ETCD_BINARY_PATH
    2. Uses ``<this file root>/bin/etcd`` if one exists
    3. Uses ``etcd`` from ``PATH``

    Usage
    ::

     server = EtcdServer("/usr/bin/etcd", 2379, "/tmp/default.etcd")
     server.start()
     client = server.get_client()
     # use client
     server.stop()

    Args:
        etcd_binary_path: path of etcd server binary (see above for fallback path)
    '''
    _port: int
    _host: str
    _etcd_binary_path: Incomplete
    _base_data_dir: Incomplete
    _etcd_cmd: Incomplete
    _etcd_proc: subprocess.Popen | None
    def __init__(self, data_dir: str | None = None) -> None: ...
    def _get_etcd_server_process(self) -> subprocess.Popen: ...
    def get_port(self) -> int:
        """Return the port the server is running on."""
    def get_host(self) -> str:
        """Return the host the server is running on."""
    def get_endpoint(self) -> str:
        """Return the etcd server endpoint (host:port)."""
    def start(self, timeout: int = 60, num_retries: int = 3, stderr: int | TextIO | None = None) -> None:
        """
        Start the server, and waits for it to be ready. When this function returns the sever is ready to take requests.

        Args:
            timeout: time (in seconds) to wait for the server to be ready
                before giving up.
            num_retries: number of retries to start the server. Each retry
                will wait for max ``timeout`` before considering it as failed.
            stderr: the standard error file handle. Valid values are
                `subprocess.PIPE`, `subprocess.DEVNULL`, an existing file
                descriptor (a positive integer), an existing file object, and
                `None`.

        Raises:
            TimeoutError: if the server is not ready within the specified timeout
        """
    def _start(self, data_dir: str, timeout: int = 60, stderr: int | TextIO | None = None) -> None: ...
    def get_client(self):
        """Return an etcd client object that can be used to make requests to this server."""
    def _wait_for_ready(self, timeout: int = 60) -> None: ...
    def stop(self) -> None:
        """Stop the server and cleans up auto generated resources (e.g. data dir)."""
