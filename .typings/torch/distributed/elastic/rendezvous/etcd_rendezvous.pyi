from _typeshed import Incomplete
from torch.distributed.elastic.rendezvous import RendezvousHandler, RendezvousParameters

__all__ = ['EtcdRendezvousRetryableFailure', 'EtcdRendezvousRetryImmediately', 'EtcdRendezvousHandler', 'EtcdRendezvous', 'create_rdzv_handler']

class EtcdRendezvousRetryableFailure(Exception): ...
class EtcdRendezvousRetryImmediately(Exception): ...

class EtcdRendezvousHandler(RendezvousHandler):
    '''
    Implements a
    :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler` interface
    backed by
    :py:class:`torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvous`.
    ``EtcdRendezvousHandler`` uses a URL to configure the type of rendezvous to
    use and to pass implementation specific configurations to the rendezvous
    module. The basic etcd rendezvous configuration URL looks like the following
    ::

     etcd://<etcd_address>:<port>/<job_id>?min_workers=<min_workers>&max_workers=<max_workers>  # noqa: W605

     -- example --

     etcd://localhost:2379/1234?min_workers=1&max_workers=3

    The URL above is interpreted as follows:

    1. Use the rendezvous handler that is registered with the ``etcd``
       scheme
    2. The ``etcd`` endpoint to use is ``localhost:2379``
    3. ``job_id == 1234`` is used as the prefix in etcd (this allows one to
       share a common etcd server for multiple jobs so long as the
       ``job_ids`` are guaranteed to be unique). Note that the job id can be
       any string (e.g. does not need to be a number) as long as it is
       unique.
    4. ``min_workers=1`` and ``max_workers=3`` specifies a range for
       membership size - Torch Distributed Elastic starts running the job as
       long as the cluster size is greater than or equal to ``min_workers``
       and admits up to ``max_workers`` into the cluster.

    Below are a full list of the parameters that can be passed to etcd
    rendezvous:

    +--------------------------------------------+--------------------------+
    | Parameter                                  | Description              |
    +============================================+==========================+
    | min_workers                                | minimum number of        |
    |                                            | workers for the          |
    |                                            | rendezvous to be valid   |
    +--------------------------------------------+--------------------------+
    | max_workers                                | maximum number of        |
    |                                            | workers to admit         |
    +--------------------------------------------+--------------------------+
    | timeout                                    | total timeout within     |
    |                                            | which next_rendezvous is |
    |                                            | expected to succeed      |
    |                                            | (default 600s)           |
    +--------------------------------------------+--------------------------+
    | last_call_timeout                          | additional wait amount   |
    |                                            | ("last call") after min  |
    |                                            | number of workers has    |
    |                                            | been reached (defaults   |
    |                                            | to 30s)                  |
    +--------------------------------------------+--------------------------+
    | etcd_prefix                                | path prefix (from etcd   |
    |                                            | root), inside which all  |
    |                                            | etcd nodes will be       |
    |                                            | created (defaults to     |
    |                                            | ``/torchelastic/p2p``)   |
    +--------------------------------------------+--------------------------+
    '''
    _rdzv_impl: Incomplete
    _local_addr: Incomplete
    def __init__(self, rdzv_impl: EtcdRendezvous, local_addr: str | None) -> None:
        """
        Args:
            rdzv_impl: the implementation of the rendezvous
            local_addr: the local address of the current node
        """
    def __del__(self) -> None: ...
    def get_backend(self) -> str: ...
    def next_rendezvous(self): ...
    def is_closed(self): ...
    def set_closed(self) -> None: ...
    def num_nodes_waiting(self): ...
    def get_run_id(self) -> str: ...
    def shutdown(self) -> bool: ...

class EtcdRendezvous:
    """A rendezvous implementation that uses `etcd <https://etcd.io/>`__ as the backend store."""
    client: Incomplete
    _prefix: Incomplete
    _run_id: Incomplete
    _num_min_workers: Incomplete
    _num_max_workers: Incomplete
    _timeout: Incomplete
    _last_call_timeout: Incomplete
    _lease_run_id_stop: Incomplete
    _lease_this_rank_stop: Incomplete
    def __init__(self, client, prefix, run_id, num_min_workers, num_max_workers, timeout, last_call_timeout) -> None: ...
    def __del__(self) -> None: ...
    _rendezvous_deadline: Incomplete
    def rendezvous_barrier(self):
        """
        Main entry point for next rendezvous.

        This method is blocking until rendezvous succeeds or a timeout occurs.

        Returns:
             ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousTimeoutError - timeout waiting for rendezvous
            RendezvousClosedError - rendezvous is or was closed while waiting
            RendezvousError - other persistent errors that
             render the rendezvous non-retryable
        """
    def init_phase(self):
        """
        Initially, the rendezvous state is expected to be one of:

        1. empty (non-existent) - in this case we try to create a new one.
        2. joinable - we try to join it.
        3. final - we announce ourselves as waiting, and go into monitoring mode

        Any other state is considered transitional, and will be retried after
        a short delay.

        Returns:
            ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousClosedError - current rendezvous was/is closed
            EtcdRendezvousRetryableFailure - observed some intermediate
             state, which is best handled by retrying later
        """
    def join_phase(self, expected_version):
        """
        We observed a rendezvous state in 'joinable' state, and attempt to join this
        particular version, and then wait for all other peers to join.
        """
    def confirm_phase(self, expected_version, this_rank):
        """
        Once the rendezvous state transitions from 'joinable' to 'frozen',
        we have every participant confirm their membership and setup per-member
        keep-alive TTL keys, and then wait for all other participants to confirm,
        which would then successfully conclude this rendezvous.
        """
    def handle_existing_rendezvous(self, expected_version) -> None:
        """
        Handle the case when there's an existing (state 'final) rendezvous already
        in place, and we have to announce ourselves waiting, and wait until
        the next rendezvous opportunity.
        """
    def try_create_rendezvous(self):
        """
        Create new rendezvous state or raise an exception that indicates an unexpected state (e.g. already exists).

        Raises:
             RendezvousError - on unexpected state
        """
    def join_rendezvous(self, expected_version):
        """Helper method for the join phase."""
    def wait_for_peers(self, expected_version):
        """Helper method for the join phase."""
    def confirm_membership(self, expected_version, this_rank):
        """Helper method for the confirm phase."""
    def wait_for_final(self, expected_version):
        """Helper method for the confirm phase."""
    def announce_self_waiting(self, expected_version):
        """
        Announce this worker is waiting (via num_workers_waiting counter) to join next
        rendezvous, but only if state and version match.
        """
    def wait_for_rendezvous_to_free(self, expected_version) -> None:
        """
        When there's an existing valid rendezvous in state 'final', we have to wait until the next opportunity to join.

        Such opportunity may come from:

        1. rendezvous state changed by someone else, in which case we unblock and retry.
        2. rendezvous becomes invalid because at least one member failed to renew their
           leased keep_alive node. We detect this, and destroy the rendezvous.
        """
    def handle_join_last_call(self, expected_version, deadline) -> None:
        """
        After we reach min number of workers, one particular worker takes on the
        responsibility of waiting an additional timeout before closing the join window.
        If the worker responsible for this fails, the rendezvous will be destroyed due
        to expiring TTL, and the other participants will re-rendezvous.

        Here we expect to see state <joinable, expected_version>
        Exit gracefully if either:

        1. state becomes <frozen, expected_version>
        2. timeout happens (reaching deadline), in which case
           we try the transition to <frozen, expected_version>

        Exit with exception otherwise.
        """
    def set_closed(self) -> None:
        """
        Mark rendezvous 'closed' for current run_id, which is used to signal other
        participants to not attempt to perform (re-)rendezvous. This is useful
        when one of the workers decides the job is complete.
        """
    def get_rdzv_state(self): ...
    def try_wait_for_state_change(self, etcd_index, timeout=None): ...
    def get_path(self, path): ...
    def create_path_if_not_exists(self, full_path, ttl=None) -> None: ...
    def setup_lease_renewal(self, full_path, ttl): ...
    def store_extra_data(self, rdzv_version, key, value) -> None: ...
    def load_extra_data(self, rdzv_version, key, timeout=None): ...
    def setup_kv_store(self, rdzv_version): ...

def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    '''
    Usage:

    ::

    rdzv_params = RendezvousParameters(
                        backend="etcd",
                        endpoint="192.168.0.42:2379",
                        run_id="123",
                        min_nodes=4,
                        max_nodes=8,
                        timeout=300,
                        last_call_timeout=30,
                        etcd_prefix="custom_prefix",
                        protocol="https",
                        cacert="/etc/kubernetes/certs/ca.crt",
                        cert="/etc/kubernetes/certs/client.crt",
                        key="/etc/kubernetes/certs/client.key")
    # -- or --
    rdzv_params = RendezvousParameters(
                        backend="etcd",
                        endpoint="192.168.0.42:2379",
                        run_id="123",
                        min_nodes=4,
                        max_nodes=8)

    etcd_rdzv_handler = create_etcd_rendezvous_handler(rdzv_params)


    Where:
        run_id - unique id for this training job instance,
        min_nodes - min number of workers expected to join the rendezvous,
        max_nodes - max number of workers allowed to join the rendezvous,
                        defaults to min_workers is not specified.
        timeout - total timeout within which next_rendezvous is expected to
                      succeed; a RendezvousTimeoutError is raised otherwise;
                      Defaults is 600 (10 minutes).
        last_call_timeout - additional wait amount ("last call") after
                            min number of workers has been reached.
                            Defaults to 30 seconds.
        etcd_prefix - path prefix (from etcd root), inside which all
                      etcd nodes will be created.
                      Default is "/torchelastic/p2p".
        protocol - http (default) or https to access etcd.
        cacert - CA cert to access etcd, only makes sense with https.
        cert - client cert to access etcd, only makes sense with https.
        key - client key to access etcd, only makes sense with https.
    '''
