import enum
import logging
import queue
import socket
from _typeshed import Incomplete
from typing import Any, Protocol
from websockets.asyncio.server import ServerConnection as ServerConnection
from websockets.http11 import Request as Request

class _SessionMessage(enum.StrEnum):
    """Text messages browsers send on /state."""
    REQUEST_CONTROL = 'request_control'
    LEAVE_QUEUE = 'leave_queue'
    FORCE_CONTROL = 'force_control'
    HEARTBEAT = 'heartbeat'

class UiSession(Protocol):
    """The headless UI as the /ui stream sees it (headless_ui.HeadlessUi).

  The UI streams to one browser at a time, the controller. A session lasts while
  that browser is connected: it starts with the browser's handshake, and its
  first messages re-send everything the browser needs to draw the UI. Every
  message is one Studio UI wire command, sent as one binary WebSocket message.

  The server thread calls these; the viewer thread runs the UI frames.
  """
    def start_session(self, handshake: bytes) -> bytes:
        """Starts a session and returns the reply to the browser's handshake.

    Args:
      handshake: The browser's first /ui message (Hello handshake).

    Raises:
      ValueError: The browser cannot talk to this viewer.
    """
    def end_session(self) -> None:
        """Ends the session; frames are no longer recorded for the browser."""
    def push_input(self, message: bytes) -> bool:
        """Queues browser input for the next frame; False if it is not input."""
    def take_messages(self) -> list[bytes]:
        """Returns the messages to send to the browser now, possibly none.

    A non-empty list ends with one draw frame, after any texture changes; the
    browser acks it by its index (see last_frame_index and _UI_ACK_PREFIX).
    """
    def last_frame_index(self) -> int:
        """Returns the index of the last draw frame take_messages() returned."""

class _WebServerFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str: ...

logger: Incomplete

class _HandshakeNoiseFilter(logging.Filter):
    """Removes handshake-failure logs from connections that simply went away.

  Handshake failure are produced by two routine cases:

  1. browsers speculatively open spare connections and close them unused
  2. a page reload on model change tears connections down mid-handshake.

  The websockets library logs each as an opening-handshake failure as at ERROR
  level with a full traceback. Raising the log level would hide real handshake
  errors too, hence a filter matching exactly these signatures.
  """
    def filter(self, record: logging.LogRecord) -> bool: ...

def bind_public_socket(host: str, port: int = 0) -> socket.socket:
    '''Binds (but does not listen on) the public HTTP listening socket.

  The socket is bound in the viewer thread up front so that bind conflicts
  surface here, synchronously, rather than inside the server thread.

  Args:
    host: Interface to bind. "::" (the default) binds a dual-stack socket that
      accepts both IPv6 and IPv4 connections, falling back to IPv4-only when the
      machine has no IPv6 support.
    port: A specific port, or 0 to take the first free port starting at
      _DEFAULT_HTTP_PORT (so several viewers can run side by side).

  Returns:
    The bound socket.

  Raises:
    RuntimeError: If the requested port (or every scanned port) is taken.
  '''

class WebServer:
    """The Web Viewer's single-port server.

  Serves the browser page, WASM, and model over HTTP; streams the headless UI at
  /ui; and broadcasts the state payload at /state on a single public port from a
  background daemon thread.  State payloads are handed over through a
  lock-protected reference with latest-wins semantics: update_state() may be
  called at any rate from the viewer thread; browsers only ever see the newest
  payload.  The server thread is started once and persists.  When the model
  changes, update_model() swaps the bytes served at /model, without any restart.
  Models that may be shown soon can be announced with add_model(): browsers
  download them in the background and switch to them without a download.

  Run the simulation with the web viewer, then visit the printed URL.
  """
    http_sock: Incomplete
    ui: Incomplete
    static_files_dir: Incomplete
    mjb_data: Incomplete
    drop_queue: Incomplete
    model_cache_bytes: Incomplete
    stop_reason: str
    def __init__(self, http_sock: socket.socket, ui: UiSession | None = None, static_files_dir: str | None = None, mjb_data: bytes | None = None, drop_queue: queue.Queue[Any] | None = None) -> None:
        """Initializes the server around a pre-bound listening socket.

    The socket is bound by the viewer (see bind_public_socket) and shared with
    the server thread, so bind failures can't occur here.

    Args:
      http_sock: The listening socket for HTTP and WebSocket traffic.
      ui: The headless UI to stream at /ui. Without it, /ui rejects browsers.
      static_files_dir: The directory containing the static files to serve.
      mjb_data: The model data to serve from /model.
      drop_queue: The queue to put dropped files onto.
    """
    def update_state(self, payload: bytes) -> None:
        """Publishes the latest state payload. Called from the viewer thread."""
    def notify_ui_frame(self) -> None:
        """Tells the /ui stream that the UI has a new frame for the browser.

    Called from the viewer thread after each HeadlessUi.end_frame(), so the
    frame goes out as soon as the browser can take it.
    """
    def update_model(self, mjb_data: bytes, crc32: int | None = None) -> None:
        """Makes ``mjb_data`` the current model, served at /model.

    The bundle also joins the cache under its crc32, so late-joining browsers
    can prefetch it and nothing evicts it while it is current.

    Args:
      mjb_data: The model bundle to serve.
      crc32: The bundle's crc32 (which must match the one stamped into state
        payloads), or None to compute it here.
    """
    def add_model(self, mjb_data: bytes, crc32: int | None = None) -> int:
        """Caches a bundle at /model?crc32=... and tells browsers to prefetch it.

    Browsers download announced bundles in announcement order; announcing a
    bundle again moves it to the end. The least recently used bundles (other
    than the current model) are dropped once the cache exceeds
    model_cache_bytes, where adding, switching to, and serving a bundle all
    count as uses.

    Args:
      mjb_data: The model bundle.
      crc32: The bundle's crc32, or None to compute it here.

    Returns:
      The bundle's crc32.
    """
    def cached_model(self, crc32: int) -> bytes | None:
        """Returns the cached bundle with this crc32, or None.

    Serving a bundle counts as a use for eviction (see add_model).
    """
    @property
    def viewer_count(self) -> int:
        """Number of browsers currently connected to the /state stream."""
    @property
    def models_generation(self) -> int:
        """Counts changes to the prefetch line; see prefetch_line()."""
    def prefetch_line(self) -> str:
        '''Returns the prefetch line for browsers; see _PREFETCH_PREFIX.

    Lists the bundles in the order to download them (see _prefetch_items); a
    bare "prefetch=" while nothing but the current model is cached.
    '''
    def set_theme(self, theme: str) -> None:
        """Records a UI theme picked by the controller browser."""
    def pop_theme(self) -> str | None:
        """Returns and clears any UI theme picked by the controller browser."""
    def set_client_camera(self, camera: bytes) -> None:
        """Records the latest camera sent by the controlling browser."""
    def take_client_camera(self) -> bytes | None:
        """Returns the camera the controller last sent, once; else None."""
    def start(self) -> None:
        """Starts the server on a background daemon thread."""
    def stop(self, reason: str = '') -> None:
        """Stops the server thread.

    Args:
      reason: Why the viewer stopped, when this is not a normal exit: a one-line
        summary of the exception that ended it. Sent to the connected browsers,
        which show it on the page.
    """
