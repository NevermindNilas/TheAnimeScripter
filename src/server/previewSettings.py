import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.infra.logAndPrint import logAndPrint


class PreviewSink:
    """
    Thread-safe holder for the latest preview JPEG.

    The writer's drain thread calls update() with each JPEG pulled off ffmpeg's
    mjpeg pipe; HTTP request handlers read the latest frame (one-shot via
    latest(), or blocking-for-next via waitNext() for the multipart stream).
    Only the most recent frame is kept -- preview is lossy-latest by design.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0

    def update(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def latest(self):
        with self._cond:
            return self._jpeg, self._seq

    def waitNext(self, lastSeq: int, timeout: float = 1.0):
        """Block until a frame newer than lastSeq is available (or timeout)."""
        with self._cond:
            if self._seq == lastSeq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq


_INDEX_HTML = b"""<!DOCTYPE html>
<html>
<head>
    <title>TAS Preview</title>
    <style>
        html, body {
            margin: 0; padding: 0; height: 100%;
            background-color: #000; overflow: hidden;
        }
        body { display: flex; align-items: center; justify-content: center; }
        img { max-width: 100vw; max-height: 100vh; width: auto; height: auto; display: block; }
    </style>
</head>
<body>
    <img id="preview" src="/stream" alt="Preview">
    <script>
        // If the multipart stream drops (e.g. encoder finished), fall back to
        // polling the one-shot endpoint so the last frame stays visible.
        const img = document.getElementById('preview');
        img.addEventListener('error', function () {
            setTimeout(function () { img.src = '/image?' + Date.now(); }, 1000);
        });
    </script>
</body>
</html>
"""


class PreviewHTTPHandler(BaseHTTPRequestHandler):
    sink: PreviewSink = None

    def _noCache(self) -> None:
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self._noCache()
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
        elif self.path.startswith("/stream"):
            self._serveStream()
        elif self.path.startswith("/image"):
            self._serveImage()
        else:
            self.send_error(404, "Not found")

    def _serveImage(self):
        jpeg, _ = self.sink.latest() if self.sink else (None, 0)
        if not jpeg:
            self.send_error(404, "Preview not available yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self._noCache()
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _serveStream(self):
        if not self.sink:
            self.send_error(404, "Preview not available")
            return
        self.send_response(200)
        self.send_header("Age", "0")
        self._noCache()
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        lastSeq = -1
        try:
            while True:
                jpeg, seq = self.sink.waitNext(lastSeq, timeout=1.0)
                # waitNext returns the current frame on timeout too, so only
                # emit when the sequence actually advanced -- otherwise an idle
                # stream would re-send the same JPEG every timeout interval.
                if jpeg is None or seq == lastSeq:
                    continue
                lastSeq = seq
                # A failed write here (client closed the tab) raises and breaks
                # the loop -- that's the intended disconnect path.
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except BrokenPipeError, ConnectionError, OSError:
            pass  # client disconnected

    def log_message(self, format, *args):
        pass


class Preview:
    def __init__(
        self,
        previewSink: PreviewSink,
        localHost: str = "127.0.0.1",
        port: int = 5000,
    ) -> None:
        self.localHost = localHost
        self.port = port
        self.previewSink = previewSink
        self.server = None
        self.serverThread = None

        PreviewHTTPHandler.sink = previewSink

    def start(self) -> None:
        try:
            # ThreadingHTTPServer: a long-lived /stream connection must not block
            # /image or a second client.
            self.server = ThreadingHTTPServer(
                (self.localHost, self.port), PreviewHTTPHandler
            )
            self.server.daemon_threads = True

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

        logAndPrint("Preview server stopped", "green")
