import logging
import os
from threading import Thread, Lock
import socketio
from flask import Flask
from flask_cors import CORS
from urllib.parse import urlparse

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("socketio").setLevel(logging.ERROR)
logging.getLogger("engineio").setLevel(logging.ERROR)

os.environ["FLASK_ENV"] = "production"


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
        with self._lock:
            self.data.update(new_data)
        sio.emit("progress_update", self.data)

    def get(self):
        with self._lock:
            return self.data.copy()


progressState = ProgressState()

sio = socketio.Server(cors_allowed_origins="*", async_mode="threading")
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
CORS(app)


@sio.event
def getProgress(sid):
    sio.emit("progress_update", progressState.get(), room=sid)


@sio.event
def shutdown(sid):
    logging.info("Received shutdown request.")
    sio.emit("server_shutdown", {"message": "Server is shutting down."}, room=sid)


@sio.event
def connect(sid, environ):
    logging.info(f"Client connected: {sid}")


def runServer(host):
    logging.info(f"Starting AE comms server on {host}...")
    import sys
    from io import StringIO

    originalSTDERR = sys.stderr
    sys.stderr = StringIO()

    parsed = urlparse(host if "://" in host else f"//{host}", scheme="http")
    hostname = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8080

    logging.info(f"AE Comms Server running on {hostname}:{port}")

    try:
        app.run(
            host=hostname, port=port, debug=False, threaded=True, use_reloader=False
        )
    finally:
        sys.stderr = originalSTDERR


def startServerInThread(host):
    serverThread = Thread(target=runServer, args=(host,))
    serverThread.daemon = True
    serverThread.start()
    return serverThread
