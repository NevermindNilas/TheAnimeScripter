"""Tests for the preview sink/sampler contract.

The preview must cost the pipeline nothing when nobody is watching, must
throttle on WALL time (not video time, or a 20x-realtime run emits 40 preview
frames a second nobody sees), and must never busy-spin an HTTP handler.
"""

import socket
import threading
import time

import pytest

previewSettings = pytest.importorskip("src.server.previewSettings")

PreviewSink = previewSettings.PreviewSink
PreviewSampler = previewSettings.PreviewSampler
Preview = previewSettings.Preview


def test_sinkHasNoViewersByDefault():
    assert PreviewSink().hasViewers() is False


def test_viewerJoinLeaveTracksDemand():
    sink = PreviewSink()
    sink.viewerJoined()
    assert sink.hasViewers() is True
    sink.viewerLeft()
    assert sink.hasViewers() is False


def test_imagePullCountsAsDemand():
    sink = PreviewSink()
    sink.update(b"jpeg")
    sink.latest()  # a /image request
    assert sink.hasViewers() is True


def test_viewerLeftNeverGoesNegative():
    # A handler whose finally-block runs twice must not leave the count below
    # zero, or a later join would be swallowed and demand gating would be stuck.
    sink = PreviewSink()
    sink.viewerLeft()
    sink.viewerLeft()
    sink.viewerJoined()
    assert sink.hasViewers() is True


def test_samplerWantsNothingWithoutViewers():
    # The whole zero-overhead claim: no viewer -> the writer never hands over a
    # frame, no matter how long the run goes.
    sampler = PreviewSampler(PreviewSink(), interval=0.0)
    assert sampler.wants() is False


def test_samplerThrottlesOnWallClock():
    sink = PreviewSink()
    sink.viewerJoined()
    sampler = PreviewSampler(sink, interval=10.0)
    assert sampler.wants() is True
    sampler.submit(_frame())
    # Interval has not elapsed, so a second frame is not wanted -- even though a
    # fast pipeline may have encoded hundreds of frames in between.
    assert sampler.wants() is False


def test_samplerServesOutputResolutionByDefault():
    # The preview shows what the encoder sees. Downscaling is not what makes the
    # sampler cheap (demand gating and the wall-clock throttle are), so it must
    # not quietly degrade the picture.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    sink = PreviewSink()
    sink.viewerJoined()
    sampler = PreviewSampler(sink, interval=0.0)
    sampler.submit(_frame(1080, 1920))

    jpeg = _waitForFrame(sink)
    assert jpeg is not None
    assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"

    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (1080, 1920)


def test_samplerDownscalesWhenCapped():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    sink = PreviewSink()
    sink.viewerJoined()
    sampler = PreviewSampler(sink, interval=0.0, maxHeight=180)
    sampler.submit(_frame(1080, 1920))

    jpeg = _waitForFrame(sink)
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[0] == 180


def test_samplerHandlesUint16():
    np = pytest.importorskip("numpy")
    sink = PreviewSink()
    sink.viewerJoined()
    sampler = PreviewSampler(sink, interval=0.0)
    sampler.submit(np.full((64, 64, 3), 4096, dtype=np.uint16))
    assert _waitForFrame(sink) is not None


def test_streamHandlerDoesNotSpinBeforeFirstFrame():
    # Regression: the handler used to seed lastSeq=-1 while the sink starts at
    # seq=0, so waitNext never parked and the handler burned 100% of a core (and
    # the GIL) for as long as a client was attached with no frame produced yet.
    psutil = pytest.importorskip("psutil")
    import os

    sink = PreviewSink()
    server = Preview(previewSink=sink, port=5077)
    server.start()
    proc = psutil.Process(os.getpid())

    sock = socket.socket()
    sock.settimeout(5)
    try:
        sock.connect(("127.0.0.1", 5077))
        sock.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
        time.sleep(0.3)  # let the handler reach its wait loop

        before = sum(proc.cpu_times()[:2])
        time.sleep(1.0)
        after = sum(proc.cpu_times()[:2])
    finally:
        sock.close()
        server.close()

    # A spinning handler burns ~1.0 CPU-second per wall-second. A parked one
    # burns ~0. Anything under a quarter core is comfortably "not spinning".
    assert (after - before) < 0.25


def _frame(h: int = 64, w: int = 64):
    np = pytest.importorskip("numpy")
    return np.zeros((h, w, 3), dtype=np.uint8)


def _waitForFrame(sink, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jpeg, _ = sink.latest()
        if jpeg is not None:
            return jpeg
        time.sleep(0.01)
    return None


def test_streamServesQueuedFrameToLateClient():
    # A client attaching after frames already flowed must get the latest one
    # immediately, not wait for the next sample.
    sink = PreviewSink()
    sink.update(b"\xff\xd8payload\xff\xd9")
    server = Preview(previewSink=sink, port=5078)
    server.start()

    sock = socket.socket()
    sock.settimeout(5)
    try:
        sock.connect(("127.0.0.1", 5078))
        sock.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
        data = b""
        deadline = time.monotonic() + 5
        while b"\xff\xd9" not in data and time.monotonic() < deadline:
            data += sock.recv(4096)
        assert b"payload" in data
    finally:
        sock.close()
        server.close()


def test_coldImageRequestWaitsForFirstFrame():
    # Demand-gating means no frame exists until something asks for one, so the
    # first /image request necessarily arrives before any frame has been made.
    # It must register demand and wait, not 404 -- a one-shot poller would
    # otherwise never see an image.
    import urllib.request

    sink = PreviewSink()
    server = Preview(previewSink=sink, port=5080)
    server.start()

    def produceOnDemand():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if sink.hasViewers():
                sink.update(b"\xff\xd8lateframe\xff\xd9")
                return
            time.sleep(0.01)

    threading.Thread(target=produceOnDemand, daemon=True).start()
    try:
        body = urllib.request.urlopen("http://127.0.0.1:5080/image", timeout=5).read()
        assert b"lateframe" in body
    finally:
        server.close()


def test_viewerCountReleasedOnDisconnect():
    sink = PreviewSink()
    server = Preview(previewSink=sink, port=5079)
    server.start()
    try:
        sock = socket.socket()
        sock.settimeout(5)
        sock.connect(("127.0.0.1", 5079))
        sock.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
        deadline = time.monotonic() + 5
        while not sink.hasViewers() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sink.hasViewers() is True

        sock.close()
        # The handler only notices the dead socket when it next writes, so push
        # frames until the viewer count drops back to zero.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            sink.update(b"\xff\xd8x\xff\xd9")
            if not sink.hasViewers():
                break
            time.sleep(0.05)
        assert sink.hasViewers() is False
    finally:
        server.close()


def test_closeStopsWorkerEvenWithAFrameQueued():
    # The single queue slot is usually occupied at end of run, so close() must
    # evict the pending frame to get its sentinel in. Otherwise the worker parks
    # on get() forever holding a full-size frame -- one leaked thread and frame
    # per video in a batch run.
    sink = PreviewSink()
    sampler = PreviewSampler(sink, interval=0.0)
    sampler._queue.put(_frame())  # occupy the slot, as a mid-encode sample would
    sampler.close()
    sampler._thread.join(timeout=5)
    assert not sampler._thread.is_alive()


def test_samplerDropsFrameWhenWorkerBusy():
    # maxsize=1: preview is lossy-latest. A backed-up worker must never make the
    # writer thread block.
    sink = PreviewSink()
    sink.viewerJoined()
    sampler = PreviewSampler(sink, interval=0.0)
    sampler._queue.put(_frame())  # occupy the single slot
    start = time.monotonic()
    sampler.submit(_frame())
    assert (time.monotonic() - start) < 0.5  # returned immediately, did not block
