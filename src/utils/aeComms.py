import logging
import os
from threading import Lock
from flask import Flask
from flask_socketio import SocketIO
from urllib.parse import urlparse
from time import time

import flask.cli as flask_cli

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("engineio").setLevel(logging.ERROR)
logging.getLogger("socketio").setLevel(logging.ERROR)
flask_cli.show_server_banner = lambda *args, **kwargs: None

os.environ["FLASK_ENV"] = "production"


socketio = None


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
        with self._lock:
            self.data.update(new_data)
        if socketio:
            socketio.emit("progress", self.data)

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


def runServer(host):
    global socketio

    logging.info(f"Starting AE comms server on {host}...")

    parsed = urlparse(host if "://" in host else f"//{host}", scheme="http")
    hostname = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8080

    # Initialize SocketIO with CORS support
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )

    @socketio.on("connect")
    def handle_connect():
        logging.info("Client connected to Socket.IO")
        socketio.emit("progress", progressState.get())

    @socketio.on("disconnect")
    def handle_disconnect():
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

        socketio.emit(
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
        socketio.emit(
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
    from threading import Thread

    serverThread = Thread(target=runServer, args=(host,))
    serverThread.daemon = True
    serverThread.start()
    return serverThread
