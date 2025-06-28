import logging
from threading import Thread, Lock
import socketio
from flask import Flask
from flask_cors import CORS


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
        sio.emit("progress_update", self.data)  # Emit update to all connected clients

    def get(self):
        with self._lock:
            return self.data.copy()


progressState = ProgressState()

sio = socketio.Server(cors_allowed_origins="*", async_mode="threading")
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
CORS(app)


@sio.event
def connect(sid, environ):
    logging.info(f"Client connected: {sid}")


@sio.event
def disconnect(sid):
    logging.info(f"Client disconnected: {sid}")


@sio.event
def getProgress(sid):
    sio.emit("progress_update", progressState.get(), room=sid)


@sio.event
def shutdown(sid):
    logging.info("Received shutdown request.")
    sio.emit("server_shutdown", {"message": "Server is shutting down."}, room=sid)


def runServer(host, port):
    logging.info(f"Starting AE comms server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


def startServerInThread(host, port):
    serverThread = Thread(target=runServer, args=(host, port))
    serverThread.daemon = True
    serverThread.start()
    return serverThread
