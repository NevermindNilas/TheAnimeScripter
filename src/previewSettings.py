import io
import numpy as np
import os
import logging
import signal

from queue import Queue
from flask import Flask, Response, jsonify
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from .coloredPrints import green

class Preview:
    def __init__(self, localHost: str = "127.0.0.1", port: int = 5000, writeBuffer: Queue = None):
        self.localHost = localHost
        self.port = port
        self.app = Flask(__name__)
        self.writeBuffer = writeBuffer
        self.readQueue = Queue()
        self.app.add_url_rule("/frame", "getFrame", self.getFrame)
        self.app.add_url_rule("/lowFrame", "getFrameLowRes", self.getFrame)
        self.app.add_url_rule("/stopServer", "stopServer", self.stopServer, methods=['GET'])
        
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.frame: np.ndarray = None
        self.lastFrame: np.ndarray = None
        self.exit = False

    def getFrame(self):
        try:
            self.frame = self.writeBuffer.peek()
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
        except Exception as e:
            print(f"Error in getFrame: {e}")
            logging.error(f"Error in getFrame: {e}")
            return Response("Error while converting frame", status=500)
        
    def getFrameLowRes(self):
        try:
            self.frame = self.writeBuffer.peek()
            if self.frame is not None:
                self.lastFrame = self.frame
    
            frameToUse = self.frame if self.frame is not None else self.lastFrame
    
            if frameToUse is None:
                return Response("No frame available", status=500)
    
            imgMode = "L" if frameToUse.shape[2] == 1 else "RGB"
            img = Image.fromarray(frameToUse, imgMode)
    
            newWidth = img.width // 4
            newHeight = img.height // 4
    
            img = img.resize((newWidth, newHeight))
    
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            return Response(buf, mimetype="image/jpeg")
        except Exception as e:
            print(f"Error in getFrameLowRes: {e}")
            logging.error(f"Error in getFrameLowRes: {e}")
            return Response("Error while converting frame", status=500)

    def stopServer(self):
        self.exit = True
        os.kill(os.getpid(), signal.SIGINT)
        return jsonify({"success": True, "message": "Server is shutting down..."})

    def start(self):
        print(green(f"Starting preview server at http://{self.localHost}:{self.port}"))
        os.environ['FLASK_ENV'] = 'production'
        self.app.logger.disabled = True
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        log.disabled = True
        self.app.run(host=self.localHost, port=self.port)

    def close(self):
        try:
            self.exit = True
            self.executor.shutdown()
            self.stopServer()
        except Exception as e:
            print(f"Error in close: {e}")
            logging.error(f"Error in close: {e}")