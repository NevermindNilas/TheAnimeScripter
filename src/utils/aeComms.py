import logging
import os
from threading import Lock
from multiprocessing import get_context
from queue import Empty
from flask import Flask
from flask_socketio import SocketIO, emit
from urllib.parse import urlparse
from time import time

import flask.cli as flask_cli

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("engineio").setLevel(logging.ERROR)
logging.getLogger("socketio").setLevel(logging.ERROR)
flask_cli.show_server_banner = lambda *args, **kwargs: None

os.environ["FLASK_ENV"] = "production"


socketio = None
progressQueue = None
serverProcess = None
connectedClients = 0
connectedClientsLock = Lock()


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

app = Flask(__name__)

EMITRATE = 25


def runServer(host, queue=None):
    global socketio
    global connectedClients

    logging.info(f"Starting AE comms server on {host}...")

    parsed = urlparse(host if "://" in host else f"//{host}", scheme="http")
    hostname = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8080

    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        manage_session=False,
        logger=False,
        engineio_logger=False,
    )

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
    def handle_connect():
        global connectedClients
        with connectedClientsLock:
            connectedClients += 1
        logging.info("Client connected to Socket.IO")
        emit("progress", progressState.get())

    @socketio.on("disconnect")
    def handle_disconnect():
        global connectedClients
        with connectedClientsLock:
            connectedClients = max(0, connectedClients - 1)
        logging.info("Client disconnected from Socket.IO")

    @socketio.on("cancel")
    def handle_cancel():
        logging.info("Cancel request received from client")

    @socketio.on("handshake")
    def handle_handshake(data):
        """
        Handle handshake from frontend.
        Responds with capabilities.
        """
        logging.info("Handshake received")

        emit(
            "handshake_ack",
            {
                "capabilities": ["cancel", "progress", "heartbeat"],
            },
        )

    @socketio.on("ping")
    def handle_ping(data):
        """
        Handle heartbeat ping from frontend.
        Responds with pong containing the original timestamp for latency calculation.
        """
        emit(
            "pong",
            {
                "timestamp": data.get("timestamp", time()),
                "serverTime": time(),
            },
        )

    logging.info(f"AE Comms Server running on {hostname}:{port}")
    socketio.run(
        app,
        host=hostname,
        port=port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


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
