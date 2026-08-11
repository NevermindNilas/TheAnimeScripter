import logging
from multiprocessing import get_context
from queue import Empty
from threading import Lock
from time import time
from urllib.parse import urlparse

socketio = None
app = None
progressQueue = None
serverProcess = None
connectedClients = 0
connectedClientsLock = Lock()


def _loadSocketIOStack():
    import socketio as sio_module

    logging.getLogger("engineio").setLevel(logging.ERROR)
    logging.getLogger("socketio").setLevel(logging.ERROR)

    return sio_module


def hasConnectedClients():
    global connectedClients
    with connectedClientsLock:
        return connectedClients > 0


class ProgressState:
    def __init__(self):
        self._lock = Lock()
        self.data = {
            "currentFrame": 0,
            "totalFrames": 1,
            "fps": 0.0,
            "eta": 0.0,
            "elapsedTime": 0.0,
            "status": "Initializing...",
        }

    def update(self, new_data):
        global socketio
        global progressQueue
        with self._lock:
            self.data.update(new_data)
            payload = self.data.copy()

        if progressQueue is not None:
            try:
                progressQueue.put_nowait(payload)
            except Exception:
                pass
            return

        if socketio and hasConnectedClients():
            socketio.emit("progress", payload)

    def get(self):
        with self._lock:
            return self.data.copy()

    def setCompleted(self, outputPath=None):
        """Emit explicit completion signal to frontend."""
        self.update(
            {
                "status": "completed",
                "outputPath": outputPath,
            }
        )
        logging.info("Processing completed, status emitted to frontend")

    def setFailed(self, error=None):
        """Emit explicit failure signal to frontend."""
        self.update(
            {
                "status": "failed",
                "error": str(error) if error else "Unknown error",
            }
        )
        logging.info(f"Processing failed: {error}")


progressState = ProgressState()


def reportTerminalStatus(processingError, outputPath, benchmark=False):
    """Send a standalone capability's final status to the After Effects panel.

    ``--segment``/``--obj_detect``/``--stabilize`` bypass ``main.py``'s
    ``start()``, and therefore its ``_notifyAdobe``, so each has to report for
    itself; without this the panel sat on the last progress string forever
    (issues #269, #236). The failure test mirrors ``main.py:_videoFailed``: an
    exception OR a missing/0-byte output counts as failed, and benchmark runs
    write no output by design so their size is not checked.
    """
    from src.io.runOutcome import outputWasWritten

    wroteOutput = benchmark or outputWasWritten(outputPath)

    if processingError is not None or not wroteOutput:
        progressState.setFailed(
            error=str(processingError)
            if processingError is not None
            else "Output file not found after processing"
        )
    else:
        progressState.setCompleted(outputPath=outputPath)


# Cache of the most recent progress payload seen by the relay. The Socket.IO
# server runs in a child process whose `progressState` is NOT the instance the
# worker updates (updates arrive over the queue), so a freshly-connected client
# must be sent this cached payload rather than the always-stale progressState.
latestProgressPayload = None

EMITRATE = 25


def _exclusiveBindProbe(hostname, port):
    """Fail fast if another process already owns ``hostname:port``.

    Werkzeug's ``make_server`` sets SO_REUSEADDR, and on Windows that lets a
    second bind of a busy port *succeed* silently: the server process stays
    alive while every connection lands on whoever bound first, so the AE panel
    sits on its initial state forever with no diagnostic (issues #269/#236's
    failure mode through a different door). SO_EXCLUSIVEADDRUSE makes the
    conflict raise; on POSIX a plain bind already raises EADDRINUSE.
    """
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((hostname, port))
    finally:
        probe.close()


def runServer(host, queue=None, bindStatus=None):
    global socketio
    global app
    global connectedClients

    logging.info(f"Starting AE comms server on {host}...")

    parsed = urlparse(host if "://" in host else f"//{host}", scheme="http")
    hostname = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8080

    # Probe before the heavy socketio imports so the parent learns the
    # outcome quickly. A conflicting bind between probe and make_server is a
    # sub-second race we accept; the deterministic busy-port case is caught.
    try:
        _exclusiveBindProbe(hostname, port)
    except OSError as e:
        logging.error(f"AE comms server cannot bind {hostname}:{port}: {e}")
        if bindStatus is not None:
            bindStatus.put(("error", f"port {port} on {hostname} is in use ({e})"))
        return
    if bindStatus is not None:
        bindStatus.put(("ok", f"{hostname}:{port}"))

    sio_module = _loadSocketIOStack()

    socketio = sio_module.Server(
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )
    app = sio_module.WSGIApp(socketio)

    def relayProgressFromQueue():
        global latestProgressPayload
        if queue is None:
            return

        lastEmit = 0.0
        minEmitInterval = 1.0 / EMITRATE
        while True:
            try:
                latestPayload = queue.get(timeout=0.1)
            except Empty:
                continue

            while True:
                try:
                    latestPayload = queue.get_nowait()
                except Empty:
                    break

            # Record the latest payload even when no client is connected, so a
            # client connecting later immediately receives the true current (or
            # terminal) state instead of a stale/empty snapshot.
            latestProgressPayload = latestPayload

            if not hasConnectedClients():
                continue

            elapsed = time() - lastEmit
            if elapsed < minEmitInterval:
                # Throttle without dropping: wait out the remaining interval so
                # the latest payload (e.g. a terminal completed/failed state) is
                # always delivered.
                socketio.sleep(minEmitInterval - elapsed)

            socketio.emit("progress", latestPayload)
            lastEmit = time()

    if queue is not None:
        socketio.start_background_task(relayProgressFromQueue)

    @socketio.on("connect")
    def handle_connect(sid, environ):
        global connectedClients
        with connectedClientsLock:
            connectedClients += 1
        logging.info("Client connected to Socket.IO")
        socketio.emit(
            "progress",
            latestProgressPayload
            if latestProgressPayload is not None
            else progressState.get(),
            to=sid,
        )

    @socketio.on("disconnect")
    def handle_disconnect(sid):
        global connectedClients
        with connectedClientsLock:
            connectedClients = max(0, connectedClients - 1)
        logging.info("Client disconnected from Socket.IO")

    @socketio.on("cancel")
    def handle_cancel(sid):
        # NOTE: not wired to the worker yet — there is no IPC path to interrupt
        # the frame loop, so this only logs. "cancel" is intentionally NOT
        # advertised in handshake capabilities until a real cancel is implemented
        # (shared multiprocessing.Event set here + polled in main.py process()).
        logging.info("Cancel request received from client (no-op: not implemented)")

    @socketio.on("handshake")
    def handle_handshake(sid, data):
        """
        Handle handshake from frontend.
        Responds with capabilities.
        """
        logging.info("Handshake received")

        socketio.emit(
            "handshake_ack",
            {
                "capabilities": ["progress", "heartbeat"],
            },
            to=sid,
        )

    @socketio.on("ping")
    def handle_ping(sid, data):
        """
        Handle heartbeat ping from frontend.
        Responds with pong containing the original timestamp for latency calculation.
        """
        socketio.emit(
            "pong",
            {
                "timestamp": data.get("timestamp", time()),
                "serverTime": time(),
            },
            to=sid,
        )

    logging.info(f"AE Comms Server running on {hostname}:{port}")

    # Werkzeug's request handler exposes ``werkzeug.socket`` in the WSGI
    # environ, which ``simple-websocket`` (used by python-engineio for
    # threading-mode async) requires to upgrade HTTP -> WebSocket. wsgiref
    # provides no such hook, so the upgrade dies with
    # "Cannot obtain socket from WSGI environment.".
    from werkzeug.serving import WSGIRequestHandler, make_server

    class QuietHandler(WSGIRequestHandler):
        def log_request(self, *args, **kwargs):
            return

        def log_message(self, format, *args):
            return

    httpd = make_server(
        hostname,
        port,
        app,
        threaded=True,
        request_handler=QuietHandler,
    )
    httpd.serve_forever()


def startServerInThread(host, bindTimeout=15.0):
    global progressQueue
    global serverProcess

    if serverProcess is not None and serverProcess.is_alive():
        return serverProcess

    context = get_context("spawn")
    progressQueue = context.Queue()
    bindStatus = context.Queue()
    serverProcess = context.Process(
        target=runServer, args=(host, progressQueue, bindStatus)
    )
    serverProcess.daemon = True
    serverProcess.start()

    # The bind happens in the child, so a busy port used to be invisible to
    # the caller's try/except: the panel just never received an event. Wait
    # for the child to report; raise so the failure is loud and actionable.
    try:
        status, detail = bindStatus.get(timeout=bindTimeout)
    except Empty:
        status = "error"
        detail = f"server process did not report readiness within {bindTimeout:.0f}s"

    if status != "ok":
        failed = serverProcess
        serverProcess = None  # don't let the dead child mask a later retry
        try:
            failed.terminate()
        except Exception:
            pass
        raise RuntimeError(f"AE comms server failed to start: {detail}")

    return serverProcess
