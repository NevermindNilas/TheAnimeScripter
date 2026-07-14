import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.infra.logAndPrint import logAndPrint


class PreviewSink:
    """
    Thread-safe holder for the latest preview JPEG, plus the demand signal the
    writer uses to decide whether producing a preview frame is worth anything.

    A preview frame nobody is looking at is pure waste, so the sink counts the
    HTTP clients currently attached to /stream and the recent one-shot /image
    pulls. hasViewers() is false when the page is closed (the common CLI case),
    and PreviewSampler then never touches a frame at all.
    """

    # A one-shot /image pull counts as demand for this long, so a poller that
    # refreshes every second keeps the sampler alive without a /stream socket.
    PULL_TTL = 3.0

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self._viewers = 0
        self._lastPull = 0.0

    def update(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def latest(self):
        with self._cond:
            self._lastPull = time.monotonic()
            return self._jpeg, self._seq

    def waitNext(self, lastSeq: int, timeout: float = 1.0):
        """
        Block until a JPEG newer than lastSeq is available, or until `timeout`.

        Callers pass the seq they last sent; the initial seq is 0, so a client
        that attaches before the first frame exists must pass 0 and will park on
        the condition instead of spinning.
        """
        with self._cond:
            if self._seq == lastSeq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq

    def viewerJoined(self) -> None:
        with self._cond:
            self._viewers += 1

    def viewerLeft(self) -> None:
        with self._cond:
            self._viewers = max(0, self._viewers - 1)

    def hasViewers(self) -> bool:
        with self._cond:
            if self._viewers > 0:
                return True
            return (time.monotonic() - self._lastPull) < self.PULL_TTL


class PreviewSampler:
    """
    Produces preview JPEGs from the frames the writer is already encoding.

    The writer calls wants() once per frame; that is two comparisons and is
    False whenever the preview page is closed or the wall-clock interval has not
    elapsed, so an unwatched preview costs the pipeline nothing. On a due frame
    the writer calls submit(), which does one host-side copy and hands off -- the
    downscale and JPEG encode happen on this class's worker thread, never on the
    writer's.

    The throttle is WALL-clock, not video-time: a pipeline running at 20x
    realtime emits the same 2 preview frames/sec as one running at 1x, because
    that is all a browser can show.

    Preview frames go out at the output resolution -- what the encoder sees is
    what you see. `maxHeight` can cap that (None = no cap), but downscaling is
    not what makes this cheap: at 2 frames/sec a full 1080p JPEG encode is 0.6%
    of one worker-thread core, so the picture is not worth degrading to save it.
    """

    def __init__(
        self,
        sink: PreviewSink,
        interval: float = 0.5,
        maxHeight: int | None = None,
        quality: int = 90,
    ) -> None:
        self.sink = sink
        self.interval = interval
        self.maxHeight = maxHeight
        self.quality = quality

        self._last = 0.0
        # maxsize=1: the sampler is lossy-latest by design. If the worker is
        # still encoding when the next frame is due, drop it rather than queue
        # work up behind a preview nobody is waiting on.
        self._queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="preview-encode"
        )
        self._thread.start()

    def wants(self) -> bool:
        return (
            self.sink.hasViewers() and (time.monotonic() - self._last) >= self.interval
        )

    def submit(self, frame) -> None:
        """
        Hand a frame to the encode worker. `frame` is HWC uint8/uint16, either a
        numpy array (already host-side) or a torch tensor (CPU or CUDA).

        Called on the writer thread, so it must stay cheap: the array the writer
        owns is reused for the next frame, hence the copy here.
        """
        self._last = time.monotonic()
        if self._queue.full():
            return
        try:
            self._queue.put_nowait(self._toHost(frame))
        except queue.Full:
            pass
        except Exception as e:
            logging.debug(f"Preview submit failed: {e}")

    def _toHost(self, frame):
        import numpy as np
        import torch

        if isinstance(frame, torch.Tensor):
            if frame.is_cuda:
                # Only when a cap is set: shrinking on the GPU keeps the
                # device->host copy at preview size instead of full size.
                frame = self._shrinkTensor(frame)
            return frame.cpu().numpy()
        return np.array(frame, copy=True)

    def _shrinkTensor(self, frame):
        import torch
        from torch.nn import functional as F

        h, w = frame.shape[0], frame.shape[1]
        tw, th = self._targetSize(w, h)
        if (tw, th) == (w, h):
            return frame
        small = F.interpolate(
            frame.permute(2, 0, 1).unsqueeze(0).float(),
            size=(th, tw),
            mode="area",
        )
        return (
            small.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 255 if frame.dtype == torch.uint8 else 65535)
            .to(frame.dtype)
            .contiguous()
        )

    def _targetSize(self, w: int, h: int):
        if self.maxHeight is None or h <= self.maxHeight:
            return w, h
        scale = self.maxHeight / h
        # Even dimensions keep JPEG chroma subsampling from padding.
        return max(2, int(w * scale) & ~1), self.maxHeight

    def _worker(self) -> None:
        import cv2
        import numpy as np

        while True:
            frame = self._queue.get()
            if frame is None:
                return
            try:
                if frame.dtype == np.uint16:
                    frame = (frame >> 8).astype(np.uint8)

                h, w = frame.shape[0], frame.shape[1]
                tw, th = self._targetSize(w, h)
                if (tw, th) != (w, h):
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

                if frame.ndim == 3 and frame.shape[2] == 4:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 3:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    bgr = frame

                ok, buf = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                if ok:
                    self.sink.update(buf.tobytes())
            except Exception as e:
                logging.debug(f"Preview encode failed: {e}")

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass


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

    # How long a cold /image request waits for the sampler to produce its first
    # frame. Frames only exist once something asks for them, so the very first
    # request necessarily arrives before one has been made; 404-ing here would
    # mean a one-shot poller never sees an image at all.
    FIRST_FRAME_WAIT = 3.0

    def _serveImage(self):
        if not self.sink:
            self.send_error(404, "Preview not available")
            return
        # latest() also registers the pull as demand, so a poller that never
        # opens /stream still keeps the sampler producing.
        jpeg, _ = self.sink.latest()
        if not jpeg:
            deadline = time.monotonic() + self.FIRST_FRAME_WAIT
            while not jpeg and time.monotonic() < deadline:
                jpeg, _ = self.sink.waitNext(0, timeout=0.5)
                self.sink.latest()  # keep the demand alive while we wait
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

        # Seed from the sink's initial seq, not -1: with -1 the "did the seq
        # advance" guard below can never be satisfied by waitNext's early return,
        # and the loop spins at 100% of a core (holding the GIL) for as long as
        # the client is attached with no frame produced yet.
        self.sink.viewerJoined()
        lastSeq = 0
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
        finally:
            self.sink.viewerLeft()

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
