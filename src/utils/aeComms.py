import logging
from threading import Lock
from multiprocessing import get_context
from queue import Empty
from urllib.parse import urlparse
from time import time


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

EMITRATE = 25


def runServer(host, queue=None):
    global socketio
    global app
    global connectedClients

    sio_module = _loadSocketIOStack()

    logging.info(f"Starting AE comms server on {host}...")

    parsed = urlparse(host if "://" in host else f"//{host}", scheme="http")
    hostname = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8080

    socketio = sio_module.Server(
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )
    app = sio_module.WSGIApp(socketio)

    def relayProgressFromQueue():
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

            if not hasConnectedClients():
                continue

            now = time()
            if now - lastEmit < minEmitInterval:
                continue

            socketio.emit("progress", latestPayload)
            lastEmit = now

    if queue is not None:
        socketio.start_background_task(relayProgressFromQueue)

    @socketio.on("connect")
    def handle_connect(sid, environ):
        global connectedClients
        with connectedClientsLock:
            connectedClients += 1
        logging.info("Client connected to Socket.IO")
        socketio.emit("progress", progressState.get(), to=sid)

    @socketio.on("disconnect")
    def handle_disconnect(sid):
        global connectedClients
        with connectedClientsLock:
            connectedClients = max(0, connectedClients - 1)
        logging.info("Client disconnected from Socket.IO")

    @socketio.on("cancel")
    def handle_cancel(sid):
        logging.info("Cancel request received from client")

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
                "capabilities": ["cancel", "progress", "heartbeat"],
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

    from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler
    from socketserver import ThreadingMixIn

    class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    class QuietHandler(WSGIRequestHandler):
        def log_message(self, format, *args):
            return

    httpd = make_server(
        hostname, port, app,
        server_class=ThreadingWSGIServer,
        handler_class=QuietHandler,
    )
    httpd.serve_forever()


def startServerInThread(host):
    global progressQueue
    global serverProcess

    if serverProcess is not None and serverProcess.is_alive():
        return serverProcess

    context = get_context("spawn")
    progressQueue = context.Queue()
    serverProcess = context.Process(target=runServer, args=(host, progressQueue))
    serverProcess.daemon = True
    serverProcess.start()
    return serverProcess
