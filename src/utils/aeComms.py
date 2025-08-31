import logging
import os
from threading import Thread, Lock, Event
from flask import Flask
from flask_cors import CORS
from urllib.parse import urlparse
from flask import jsonify
from flask import Response, stream_with_context
import json
import flask.cli as flask_cli

logging.getLogger("werkzeug").setLevel(logging.ERROR)

os.environ["FLASK_ENV"] = "production"


class ProgressState:
    def __init__(self):
        self._lock = Lock()
        self._event = Event()
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
        self._event.set()

    def get(self):
        with self._lock:
            return self.data.copy()

    def wait_for_update(self, timeout=None):
        self._event.wait(timeout)
        self._event.clear()


progressState = ProgressState()

app = Flask(__name__)
CORS(app)


@app.route("/progress", methods=["GET"])
def get_progress():
    return jsonify(progressState.get())


@app.route("/progress/stream")
def progress_stream():
    def eventStream():
        lastData = None
        while True:
            progressState.wait_for_update()
            data = progressState.get()
            if data != lastData:
                yield f"data: {json.dumps(data)}\n\n"
                lastData = data.copy()

    return Response(stream_with_context(eventStream()), mimetype="text/event-stream")


def runServer(host):
    logging.info(f"Starting AE comms server on {host}...")

    parsed = urlparse(host if "://" in host else f"//{host}", scheme="http")
    hostname = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8080

    logging.info(f"AE Comms Server running on {hostname}:{port}")
    flask_cli.show_server_banner = lambda *args, **kwargs: None
    app.run(host=hostname, port=port, debug=False, threaded=True, use_reloader=False)


def startServerInThread(host):
    serverThread = Thread(target=runServer, args=(host,))
    serverThread.daemon = True
    serverThread.start()
    return serverThread
