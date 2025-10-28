import numpy as np
import os
import logging
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from src.utils.logAndPrint import logAndPrint


class PreviewHTTPHandler(SimpleHTTPRequestHandler):
    previewPath = None

    def do_GET(self):
        if self.previewPath and os.path.exists(self.previewPath):
            try:
                with open(self.previewPath, "rb") as f:
                    content = f.read()

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(content))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"Error reading preview: {e}")
        else:
            self.send_error(404, "Preview not available yet")

    def log_message(self, format, *args):
        pass


class Preview:
    def __init__(
        self, localHost: str = "127.0.0.1", port: int = 5000, previewPath: str = None
    ) -> None:
        self.localHost = localHost
        self.port = port
        self.previewPath = previewPath
        self.server = None
        self.serverThread = None

        PreviewHTTPHandler.previewPath = previewPath

    def start(self) -> None:
        try:
            self.server = HTTPServer((self.localHost, self.port), PreviewHTTPHandler)

            logAndPrint(
                f"Preview URL: http://{self.localHost}:{self.port}/",
                "green",
            )

            self.serverThread = threading.Thread(
                target=self.server.serve_forever, daemon=True
            )
            self.serverThread.start()

        except Exception as e:
            logging.error(f"Preview server error: {e}")

    def close(self) -> None:
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception as e:
                logging.warning(f"Error stopping preview server: {e}")

        if self.serverThread and self.serverThread.is_alive():
            self.serverThread.join(timeout=2)

        if self.previewPath and os.path.exists(self.previewPath):
            try:
                os.remove(self.previewPath)
            except Exception:
                pass

        logAndPrint("Preview server stopped", "green")

    def add(self, frame: np.ndarray) -> None:
        pass
