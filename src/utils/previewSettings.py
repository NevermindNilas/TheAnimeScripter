import io
import numpy as np
import os
import logging
import threading
from queue import Queue, Full, Empty
from flask import Flask, Response, jsonify
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from werkzeug.serving import make_server
from .coloredPrints import green


class Preview:
    def __init__(self, local_host: str = "127.0.0.1", port: int = 5000) -> None:
        self.local_host = local_host
        self.port = port
        self.app = Flask(__name__)
        self.read_queue: Queue = Queue(maxsize=1)
        self.app.add_url_rule("/frame", "getFrame", self.getFrame)
        self.app.add_url_rule(
            "/stopServer", "stopServer", self.stopServer, methods=["GET"]
        )

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.frame: np.ndarray = None
        self.lastFrame: np.ndarray = None
        self.exit = False
        self.server = None
        self.server_thread = None

    def add(self, frame: np.ndarray) -> None:
        try:
            self.read_queue.put(frame, block=False)
        except Full:
            pass

    def getFrame(self) -> Response:
        try:
            self.frame = self.read_queue.get_nowait()
            if self.frame is not None:
                self.lastFrame = self.frame

            frameToUse = self.frame if self.frame is not None else self.lastFrame

            if frameToUse is None:
                return Response("No frame available", status=500)

            imgMode = "L" if frameToUse.shape[2] == 1 else "RGB"
            img = Image.fromarray(frameToUse, imgMode)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            return Response(buf, mimetype="image/jpeg")
        except Empty:
            if self.lastFrame is not None:
                imgMode = "L" if self.lastFrame.shape[2] == 1 else "RGB"
                img = Image.fromarray(self.lastFrame, imgMode)
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                return Response(buf, mimetype="image/jpeg")
            return Response("No frame available", status=500)
        except Exception as e:
            logging.error(f"Error in getFrame: {e}")
            return Response("Error while converting frame", status=500)

    def stopServer(self) -> Response:
        self.exit = True
        threading.Thread(target=self.close).start()
        return jsonify({"success": True, "message": "Server is shutting down..."})

    def start(self) -> None:
        print(
            green(
                f"Starting preview server at: http://{self.local_host}:{self.port}/frame"
            )
        )
        os.environ["FLASK_ENV"] = "production"
        self.app.logger.disabled = True
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log.disabled = True

        self.server = make_server(self.local_host, self.port, self.app)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.start()

    def close(self) -> None:
        self.exit = True
        if self.server:
            self.server.shutdown()
        if self.server_thread:
            self.server_thread.join()
        self.executor.shutdown(wait=True)
        print(green("Preview server has been shut down."))
