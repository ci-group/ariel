from . import _etcd_stub as etcd
from .api import RendezvousConnectionError as RendezvousConnectionError, RendezvousParameters as RendezvousParameters, RendezvousStateError as RendezvousStateError
from .dynamic_rendezvous import RendezvousBackend as RendezvousBackend, Token as Token
from .etcd_store import EtcdStore as EtcdStore
from .utils import parse_rendezvous_endpoint as parse_rendezvous_endpoint
from torch.distributed import Store as Store

class EtcdRendezvousBackend(RendezvousBackend):
    """Represents an etcd-based rendezvous backend.

    Args:
        client:
            The ``etcd.Client`` instance to use to communicate with etcd.
        run_id:
            The run id of the rendezvous.
        key_prefix:
            The path under which to store the rendezvous state in etcd.
        ttl:
            The TTL of the rendezvous state. If not specified, defaults to two hours.
    """
    _DEFAULT_TTL: int
    _client: etcd.Client
    _key: str
    _ttl: int
    def __init__(self, client: etcd.Client, run_id: str, key_prefix: str | None = None, ttl: int | None = None) -> None: ...
    @property
    def name(self) -> str:
        """See base class."""
    def get_state(self) -> tuple[bytes, Token] | None:
        """See base class."""
    def set_state(self, state: bytes, token: Token | None = None) -> tuple[bytes, Token, bool] | None:
        """See base class."""
    def _decode_state(self, result: etcd.EtcdResult) -> tuple[bytes, Token]: ...

def _create_etcd_client(params: RendezvousParameters) -> etcd.Client: ...
def create_backend(params: RendezvousParameters) -> tuple[EtcdRendezvousBackend, Store]:
    '''Create a new :py:class:`EtcdRendezvousBackend` from the specified parameters.

    +--------------+-----------------------------------------------------------+
    | Parameter    | Description                                               |
    +==============+===========================================================+
    | read_timeout | The read timeout, in seconds, for etcd operations.        |
    |              | Defaults to 60 seconds.                                   |
    +--------------+-----------------------------------------------------------+
    | protocol     | The protocol to use to communicate with etcd. Valid       |
    |              | values are "http" and "https". Defaults to "http".        |
    +--------------+-----------------------------------------------------------+
    | ssl_cert     | The path to the SSL client certificate to use along with  |
    |              | HTTPS. Defaults to ``None``.                              |
    +--------------+-----------------------------------------------------------+
    | ssl_cert_key | The path to the private key of the SSL client certificate |
    |              | to use along with HTTPS. Defaults to ``None``.            |
    +--------------+-----------------------------------------------------------+
    | ca_cert      | The path to the rool SSL authority certificate. Defaults  |
    |              | to ``None``.                                              |
    +--------------+-----------------------------------------------------------+
    '''
