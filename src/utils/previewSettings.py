import io
import numpy as np
import os
import logging
import threading
from flask import Flask, Response, jsonify
from PIL import Image
from werkzeug.serving import make_server
from src.utils.logAndPrint import logAndPrint


class Preview:
    def __init__(self, local_host: str = "127.0.0.1", port: int = 5000) -> None:
        self.local_host = local_host
        self.port = port
        self.app = Flask(__name__)
        self.app.add_url_rule("/frame", "getFrame", self.getFrame)
        self.app.add_url_rule(
            "/stopServer", "stopServer", self.stopServer, methods=["GET"]
        )

        self.frame = None
        self.lastFrame = None
        self.frameLock = threading.Lock()
        self.exit = False
        self.server = None
        self.serverThread = None

    def add(self, frame: np.ndarray) -> None:
        with self.frameLock:
            self.frame = frame
            if frame is not None:
                self.lastFrame = frame

    def getFrame(self) -> Response:
        try:
            with self.frameLock:
                frameToUse = self.frame if self.frame is not None else self.lastFrame
                self.frame = None  # Reset current frame after serving

            if frameToUse is None:
                return Response("No frame available", status=500)

            # Handle both grayscale and RGB images
            if len(frameToUse.shape) == 2 or (
                len(frameToUse.shape) == 3 and frameToUse.shape[2] == 1
            ):
                imgMode = "L"
                if (
                    len(frameToUse.shape) == 3
                ):  # Convert to 2D if it's a 3D array with 1 channel
                    frameToUse = frameToUse.squeeze(2)
            else:
                imgMode = "RGB"

            img = Image.fromarray(frameToUse, imgMode)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            return Response(buf, mimetype="image/jpeg")
        except Exception as e:
            logging.error(f"Error in getFrame: {e}")
            return Response(f"Error while converting frame: {e}", status=500)

    def stopServer(self) -> Response:
        threading.Thread(target=self.close).start()
        return jsonify({"success": True, "message": "Server is shutting down..."})

    def start(self) -> None:
        logAndPrint(
            f"Starting preview server at: http://{self.local_host}:{self.port}/frame",
            "green",
        )
        os.environ["FLASK_ENV"] = "production"
        self.app.logger.disabled = True
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log.disabled = True

        self.server = make_server(self.local_host, self.port, self.app)
        self.serverThread = threading.Thread(target=self.server.serve_forever)
        self.serverThread.daemon = True  # Exit when main thread exits
        self.serverThread.start()

    def close(self) -> None:
        self.exit = True
        if self.server:
            self.server.shutdown()
        if self.serverThread and self.serverThread.is_alive():
            self.serverThread.join(timeout=5)  # Wait up to 5 seconds

        logAndPrint("Preview server stopped", "green")
