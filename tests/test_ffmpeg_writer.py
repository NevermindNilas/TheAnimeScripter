import importlib
import importlib.util
import os
import sys
import types


def _installFakeTorch(monkeypatch):
    if importlib.util.find_spec("torch") is not None:
        return

    functionalModule = types.ModuleType("torch.nn.functional")
    nnModule = types.ModuleType("torch.nn")
    nnModule.functional = functionalModule

    fakeTorch = types.ModuleType("torch")
    fakeTorch.__version__ = "test"
    fakeTorch.uint8 = "uint8"
    fakeTorch.uint16 = "uint16"
    fakeTorch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fakeTorch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False, is_built=lambda: False),
        cudnn=types.SimpleNamespace(benchmark=False, enabled=False),
    )
    fakeTorch.device = lambda name: name
    fakeTorch.nn = nnModule

    monkeypatch.setitem(sys.modules, "torch", fakeTorch)
    monkeypatch.setitem(sys.modules, "torch.nn", nnModule)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functionalModule)


def testWriteBufferInitialNoneSentinelExitsCleanly(monkeypatch, tmp_path):
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.WriteBuffer(output=str(tmp_path / "out.mp4"))
    wb.writeBuffer.put(None)

    wb()


def testNeluxWriteBufferBenchmarkConsumesFramesWithoutEncoder(monkeypatch, tmp_path):
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        benchmark=True,
    )
    wb.writeBuffer.put(object())
    wb.writeBuffer.put(None)

    wb()

    assert wb.writtenFrames == 1
    assert wb.encoder is None


def testNeluxWriteBufferBuildsEncoderAtOutputScaleDims(monkeypatch, tmp_path):
    """--output_scale rides nelux's encoder-side resize (>= 0.18.0): the
    encoder is built at the target dims with resize=True/bilinear, and the
    pipeline keeps handing it full-res frames."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        width=1920,
        height=1080,
        output_scale_width=1280,
        output_scale_height=720,
    )

    assert (wb.outputWidth, wb.outputHeight) == (1280, 720)
    assert wb.encoderKwargs["resize"] is True
    assert wb.encoderKwargs["resize_filter"] == "bilinear"
    # The queue side is untouched: frames still enter at pipeline resolution.
    assert (wb.width, wb.height) == (1920, 1080)


def testNeluxWriteBufferWithoutOutputScaleDoesNotAskForResize(monkeypatch, tmp_path):
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        width=1920,
        height=1080,
    )

    assert (wb.outputWidth, wb.outputHeight) == (1920, 1080)
    assert "resize" not in wb.encoderKwargs


NELUX_TWINS = [
    ("x264_nelux", "x264"),
    ("slow_x264_nelux", "slow_x264"),
    ("x264_10bit_nelux", "x264_10bit"),
    ("x264_animation_nelux", "x264_animation"),
    ("x264_animation_10bit_nelux", "x264_animation_10bit"),
    ("x265_nelux", "x265"),
    ("slow_x265_nelux", "slow_x265"),
    ("x265_10bit_nelux", "x265_10bit"),
    ("av1_nelux", "av1"),
    ("slow_av1_nelux", "slow_av1"),
    ("nvenc_h264_nelux", "nvenc_h264"),
    ("slow_nvenc_h264_nelux", "slow_nvenc_h264"),
    ("nvenc_h265_nelux", "nvenc_h265"),
    ("slow_nvenc_h265_nelux", "slow_nvenc_h265"),
    ("nvenc_h265_10bit_nelux", "nvenc_h265_10bit"),
    ("nvenc_av1_nelux", "nvenc_av1"),
    ("slow_nvenc_av1_nelux", "slow_nvenc_av1"),
    ("vp9_nelux", "vp9"),
    ("prores_nelux", "prores"),
    ("gif_nelux", "gif"),
    ("lossless_nelux", "lossless"),
    ("lossless_nvenc_nelux", "lossless_nvenc"),
]


def testNeluxWriterMergesColourOptionsUnderMappingOptions(monkeypatch, tmp_path):
    """The Nelux writer mirrors colorSpaceFilter through encoder AVOptions:
    bt709 conversion+tags by default, without clobbering the mapping's own
    quality options."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    monkeypatch.setattr("src.constants.METADATAPATH", "", raising=False)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        encode_method="slow_x264_nelux",
    )
    options = wb.encoderKwargs["options"]
    assert options["colorspace"] == "bt709"
    assert options["color_primaries"] == "bt709"
    assert options["color_trc"] == "bt709"
    assert options["color_range"] == "tv"
    # The mapping's own knobs survive the merge.
    assert options["tune"] == "animation"
    assert options["g"] == "240"
    assert wb.encoderKwargs["preset"] == "slow"


def testNeluxWriterTagsBt2020FromProbedMetadata(monkeypatch, tmp_path):
    """A probed BT.2020 source converts and tags bt2020nc with the source's
    own transfer, mirroring bt2020Filter -- instead of riding the bt709 arm."""
    import json

    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    metadataPath = tmp_path / "metadata.json"
    metadataPath.write_text(
        json.dumps({"metadata": {"ColorSpace": "bt2020nc", "ColorTRT": "arib-std-b67"}})
    )
    monkeypatch.setattr("src.constants.METADATAPATH", str(metadataPath), raising=False)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        encode_method="x265_nelux",
    )
    options = wb.encoderKwargs["options"]
    assert options["colorspace"] == "bt2020nc"
    assert options["color_primaries"] == "bt2020"
    assert options["color_trc"] == "arib-std-b67"


def testNeluxWriterStillWarnsAndFallsBackOnAnUnmappedMethod(monkeypatch, tmp_path):
    """An unmapped name must stay loud (nvenc_h264 fallback + warning), not
    become a KeyError now that the mapping lives in matchNeluxEncoder."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    monkeypatch.setattr("src.constants.METADATAPATH", "", raising=False)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        encode_method="qsv_h264_nelux",
    )
    assert wb.encoderKwargs["codec"] == "h264_nvenc"


def testWriteBufferFallsBackFromNeluxMethodToItsFFmpegTwin(monkeypatch, tmp_path):
    """Every depth backend builds WriteBuffer directly, so a *_nelux
    --encode_method lands on the FFmpeg writer. matchEncoder has no arm for
    those names and returned [], leaving the command with no -c:v at all:
    FFmpeg then encoded with the container default at its own CRF and the
    requested encoder was silently lost."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    monkeypatch.setattr("src.constants.FFMPEGPATH", "ffmpeg", raising=False)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    for index, (neluxMethod, twin) in enumerate(NELUX_TWINS):
        wb = ffmpegSettings.WriteBuffer(
            output=str(tmp_path / f"out{index}.mp4"),
            encode_method=neluxMethod,
            width=64,
            height=64,
            fps=24.0,
        )
        assert wb.encode_method == twin
        # The symptom was the built command, not the attribute: assert on it.
        assert "-c:v" in wb.encodeSettings(), f"{neluxMethod} produced no -c:v"


def testWriteBufferLeavesNonNeluxMethodsAlone(monkeypatch, tmp_path):
    """The strip is suffix-matched, so pin that it touches nothing else."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    for method in (
        "x264",
        "slow_x265",
        "lossless_nvenc",
        "prores_segment",
        "png",
        "jpeg",
    ):
        wb = ffmpegSettings.WriteBuffer(
            output=str(tmp_path / "out.mp4"), encode_method=method
        )
        assert wb.encode_method == method


def testSingleImageOutputGetsItsFormatExtension(monkeypatch, tmp_path):
    """An extensionless single-image output is completed with the extension of
    the requested sequence format, not hardcoded .png."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    for method, ext in (("png", ".png"), ("jpeg", ".jpg")):
        wb = ffmpegSettings.WriteBuffer(
            output=str(tmp_path / "frame"),
            encode_method=method,
            single_image_output=True,
        )
        assert wb.output.endswith(f"frame{ext}"), method
        assert wb._shouldUseDirectImageSingleFrame(), method


def testSegmentStillEncodesProresWithANeluxMethod(monkeypatch, tmp_path):
    """animeSegment passes transparent=True, and getPixFMT rewrites the method
    to prores_segment before matchEncoder sees it -- so segment never lost its
    -c:v and must not be told a substitute encoder it will not use."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    monkeypatch.setattr("src.constants.FFMPEGPATH", "ffmpeg", raising=False)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.WriteBuffer(
        output=str(tmp_path / "out.mov"),
        encode_method="x264_nelux",
        transparent=True,
        width=64,
        height=64,
        fps=24.0,
    )
    command = wb.encodeSettings()
    assert "prores_ks" in command
    assert "libx264" not in command


def _hwUploadFormat(ffmpegSettings, tmp_path, method, bitDepth, grayscale=True):
    wb = ffmpegSettings.WriteBuffer(
        output=str(tmp_path / "out.mp4"),
        encode_method=method,
        width=64,
        height=64,
        fps=24.0,
        grayscale=grayscale,
        bitDepth=bitDepth,
    )
    command = wb.encodeSettings()
    filters = command[command.index("-vf") + 1] if "-vf" in command else ""
    if "format=p010le" in filters:
        return "p010le"
    if "format=nv12" in filters:
        return "nv12"
    return None


def testTenBitNvencUploadsP010leNotNv12(monkeypatch, tmp_path):
    """hwupload_cuda uploads whatever software format precedes it and the
    command carries no -pix_fmt, so pinning nv12 handed the encoder 8-bit
    surfaces: --depth --bit_depth 16bit on nvenc_h265 wrote an 8-bit depth map
    (and nvenc_h265_10bit tagged it Main 10 anyway). CUDA has no yuv444p10le
    hwframe format, so 4:4:4 still subsamples -- but the bit depth is kept."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    monkeypatch.setattr("src.constants.FFMPEGPATH", "ffmpeg", raising=False)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    for method in ("nvenc_h265", "slow_nvenc_h265", "nvenc_av1", "slow_nvenc_av1"):
        assert _hwUploadFormat(ffmpegSettings, tmp_path, method, "16bit") == "p010le"
        assert _hwUploadFormat(ffmpegSettings, tmp_path, method, "8bit") == "nv12"


def testH264NvencNeverGetsP010le(monkeypatch, tmp_path):
    """h264_nvenc cannot encode 10-bit -- FFmpeg fails the device with
    "Provided device doesn't support required NVENC features". getPixFMT
    forces "nvenc_h264" back to 8-bit, but "lossless_nvenc" also runs on
    h264_nvenc and skips that branch, so the gate is on the encoder name."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    monkeypatch.setattr("src.constants.FFMPEGPATH", "ffmpeg", raising=False)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    for method in (
        "nvenc_h264",
        "slow_nvenc_h264",
        "lossless_nvenc",
        "lossless_nvenc_h264",
    ):
        for bitDepth in ("8bit", "16bit"):
            assert (
                _hwUploadFormat(ffmpegSettings, tmp_path, method, bitDepth) == "nv12"
            ), f"{method} at {bitDepth} must stay 8-bit nv12"


def testCreateWriteBufferRoutesNeluxMethodsToTheNeluxWriter(monkeypatch, tmp_path):
    """By the time a writer is built the CLI has already swapped out any
    *_nelux method the run cannot honor, so this stays a plain routing check."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    for neluxMethod, twin in NELUX_TWINS:
        wb = ffmpegSettings.createWriteBuffer(
            output=str(tmp_path / "out.mp4"), encode_method=neluxMethod
        )
        assert isinstance(wb, ffmpegSettings.NeluxWriteBuffer)

        wb = ffmpegSettings.createWriteBuffer(
            output=str(tmp_path / "out.mp4"), encode_method=twin
        )
        assert isinstance(wb, ffmpegSettings.WriteBuffer)


def testMovOutputWithOpusAudioRunsTheFFmpegTwin(monkeypatch, tmp_path):
    """MOV's query_codec accepts opus/vorbis, so nelux stream-copies and then
    dies at header write ("opus only supported in MP4") -- poisoning the whole
    encode for a 0-byte .mov. The FFmpeg twin transcodes those to AAC, so the
    factory routes such runs there, per input file."""
    _installFakeTorch(monkeypatch)

    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")

    for audioCodec, expected in (
        ("opus", "WriteBuffer"),
        ("vorbis", "WriteBuffer"),
        ("aac", "NeluxWriteBuffer"),
    ):
        fakeNelux = types.SimpleNamespace(
            probe=lambda path, codec=audioCodec: {"audio_codec": codec}
        )
        monkeypatch.setitem(sys.modules, "nelux", fakeNelux)
        ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")
        monkeypatch.setattr(ffmpegSettings, "nelux", fakeNelux, raising=False)
        monkeypatch.setattr("src.constants.AUDIO", True, raising=False)

        wb = ffmpegSettings.createWriteBuffer(
            input=str(src),
            output=str(tmp_path / "out.mov"),
            encode_method="prores_nelux",
        )
        assert type(wb).__name__ == expected, audioCodec
        if expected == "WriteBuffer":
            assert wb.encode_method == "prores"

    # A non-.mov output keeps the Nelux writer even with opus audio: MP4/M4V
    # and webm both accept it (webm through allow_transcode).
    fakeNelux = types.SimpleNamespace(probe=lambda path: {"audio_codec": "opus"})
    monkeypatch.setitem(sys.modules, "nelux", fakeNelux)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")
    monkeypatch.setattr(ffmpegSettings, "nelux", fakeNelux, raising=False)
    wb = ffmpegSettings.createWriteBuffer(
        input=str(src),
        output=str(tmp_path / "out.mp4"),
        encode_method="x264_nelux",
    )
    assert type(wb).__name__ == "NeluxWriteBuffer"


def testEveryWriterCreatesItsOutputDirectory(monkeypatch, tmp_path):
    """FFmpeg's avio does not create the folder it writes into: it dies at
    open with "Error opening output <path>: No such file or directory" and
    takes the run with it (broken pipe on the first frame, 0-byte output).
    nelux's VideoEncoder does create it, but that is the dependency's
    behaviour and not something to build on across pinned versions -- pin the
    invariant on TAS's side so both writers answer for their own output."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    for method in ("x264", *(nelux for nelux, _ in NELUX_TWINS)):
        target = tmp_path / method / "TAS-Output" / "out.mp4"
        assert not target.parent.exists()

        wb = ffmpegSettings.createWriteBuffer(output=str(target), encode_method=method)

        assert target.parent.is_dir(), f"{method} left its output folder missing"
        assert os.path.normpath(wb.output) == os.path.normpath(str(target))


def _recordingWriterRegistry(monkeypatch, ffmpegSettings):
    """Swap the live-writer registry for a fresh one that records every add().

    Both writers leave it empty once they are done -- the Nelux one discards
    itself in its finally -- so a plain "is it empty afterwards" check cannot
    tell "never registered" from "registered and already gone".
    """
    import weakref

    class RecordingRegistry(weakref.WeakSet):
        def __init__(self):
            super().__init__()
            self.added = []

        def add(self, item):
            self.added.append(type(item).__name__)
            super().add(item)

    registry = RecordingRegistry()
    monkeypatch.setattr(ffmpegSettings, "_LIVE_NELUX_WRITERS", registry)
    return registry


def _uint8HwcFrame(monkeypatch):
    """A frame the Nelux encode loop takes on its HWC-uint8 fast path, under
    real torch and under this file's stub alike (the stub has no Tensor type)."""
    if importlib.util.find_spec("torch") is not None:
        import torch

        return torch.zeros((4, 4, 3), dtype=torch.uint8)

    class FakeFrame:
        ndim = 3
        dtype = "uint8"
        shape = (4, 4, 3)

        def is_contiguous(self):
            return True

    monkeypatch.setattr(sys.modules["torch"], "Tensor", FakeFrame, raising=False)
    return FakeFrame()


def testFinalizeLiveWritersReturnsTrueImmediatelyWhenNothingIsLive(
    monkeypatch, tmp_path
):
    """A Ctrl-C on the x264 path (or before any writer exists) must not spend
    the handler's deadline on an empty registry before os._exit."""
    import time

    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")
    _recordingWriterRegistry(monkeypatch, ffmpegSettings)

    start = time.monotonic()
    assert ffmpegSettings.finalizeLiveWriters(timeout=5.0) is True
    assert time.monotonic() - start < 1.0


def testFinalizeLiveWritersStopsAParkedWriterAndClosesItsEncoderOnce(
    monkeypatch, tmp_path
):
    """Cancelling a *_nelux render left a multi-hundred-MB file with no moov
    atom: the encoder runs in this process and finalizes only in its worker
    thread's finally, which os._exit(130) never runs. The writer is parked in
    its encode loop, so the handshake has to reach it from outside -- _stopNow
    breaks the loop, the finally writes the trailer, _finalized reports it."""
    import threading
    import time

    _installFakeTorch(monkeypatch)
    closes = []

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_frame(self, frame):
            pass

        def close(self):
            closes.append(1)

    fakeNelux = types.SimpleNamespace(VideoEncoder=FakeEncoder)
    monkeypatch.setitem(sys.modules, "nelux", fakeNelux)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")
    monkeypatch.setattr(ffmpegSettings, "nelux", fakeNelux, raising=False)
    monkeypatch.setattr("src.constants.METADATAPATH", "", raising=False)
    monkeypatch.setattr("src.constants.AUDIO", False, raising=False)
    registry = _recordingWriterRegistry(monkeypatch, ffmpegSettings)

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        encode_method="x264_nelux",
        width=4,
        height=4,
    )
    wb.write(_uint8HwcFrame(monkeypatch))

    worker = threading.Thread(target=wb, daemon=True)
    worker.start()
    deadline = time.monotonic() + 10.0
    while wb.writtenFrames == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert wb.writtenFrames == 1, "writer never reached its encode loop"
    assert registry.added == ["NeluxWriteBuffer"]

    assert ffmpegSettings.finalizeLiveWriters(timeout=5.0) is True
    assert wb._stopNow.is_set()
    assert wb._finalized.is_set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    # Exactly once: the trailer is written by the worker's finally and nothing
    # else, so a second close from the cancel path would be a double-free.
    assert closes == [1]


def testFinalizeLiveWritersGivesUpWithinItsTimeout(monkeypatch, tmp_path):
    """The regression that matters most. A writer can be parked inside a native
    encode_frame that no Python-level flag preempts, so this wait has to be
    bounded: an unbounded one would reintroduce the very hang os._exit exists
    to dodge, leaving the user on Ctrl-C with a process that never leaves."""
    import threading
    import time

    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")
    registry = _recordingWriterRegistry(monkeypatch, ffmpegSettings)

    class StuckWriter:
        """A writer wedged in native code: it accepts the stop request but its
        trailer never gets written, so waitFinalized always times out."""

        def __init__(self):
            self._stopNow = threading.Event()
            self._finalized = threading.Event()  # never set

        def requestStop(self):
            self._stopNow.set()

        def waitFinalized(self, timeout):
            return self._finalized.wait(timeout)

    stuck = [StuckWriter(), StuckWriter()]
    for writer in stuck:
        registry.add(writer)

    start = time.monotonic()
    assert ffmpegSettings.finalizeLiveWriters(timeout=0.25) is False
    elapsed = time.monotonic() - start
    # The deadline is shared across writers, so two wedged ones still cost one
    # timeout, not one each.
    assert elapsed < 2.0, f"waited {elapsed:.2f}s for a 0.25s deadline"
    assert all(writer._stopNow.is_set() for writer in stuck)


def testFinalizeLiveWritersNeverRaisesOnABrokenRegistryEntry(monkeypatch, tmp_path):
    """It runs on the way to os._exit(130), so it answers for its own failures:
    an exception escaping here would skip the exit and change the process exit
    code that batch callers and the AE panel read."""
    import threading

    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")
    registry = _recordingWriterRegistry(monkeypatch, ffmpegSettings)

    class BrokenWriter:
        def __init__(self):
            self._stopNow = threading.Event()

        def requestStop(self):
            self._stopNow.set()

        def waitFinalized(self, timeout):
            raise RuntimeError("registry entry is garbage")

    class UnstoppableWriter:
        """Raises on the stop request itself -- the earlier of the two loops."""

        def requestStop(self):
            raise RuntimeError("registry entry is garbage")

        def waitFinalized(self, timeout):
            return True

    # Bound to locals: the registry is a WeakSet and would drop them otherwise.
    broken = BrokenWriter()
    unstoppable = UnstoppableWriter()
    registry.add(broken)
    registry.add(unstoppable)

    assert ffmpegSettings.finalizeLiveWriters(timeout=0.25) is False


def testOneBadEntryStillLetsHealthyWritersFinalize(monkeypatch, tmp_path):
    """A writer that raises on requestStop must not abort the stop loop and
    leave the healthy writers un-asked -- they would then just time out."""
    import threading

    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")
    registry = _recordingWriterRegistry(monkeypatch, ffmpegSettings)

    class Healthy:
        def __init__(self):
            self.stopped = threading.Event()

        def requestStop(self):
            self.stopped.set()

        def waitFinalized(self, timeout):
            return self.stopped.is_set()

    class Poisoned:
        def requestStop(self):
            raise RuntimeError("garbage")

        def waitFinalized(self, timeout):
            return False

    # Bind both to locals: the registry is a WeakSet, so an entry added inline
    # is collected before finalizeLiveWriters ever sees it.
    healthy = Healthy()
    poisoned = Poisoned()
    registry.add(poisoned)
    registry.add(healthy)

    assert ffmpegSettings.finalizeLiveWriters(timeout=0.25) is False
    assert healthy.stopped.is_set(), "the bad entry swallowed the healthy one's stop"


def testPlainFFmpegWriteBufferIsNeverRegisteredAsALiveWriter(monkeypatch, tmp_path):
    """Only in-process encoders belong in the registry. WriteBuffer's close()
    waits on the FFmpeg subprocess -- exactly the join os._exit exists to skip
    -- and its child finalizes the container itself on stdin EOF, so a partial
    x264 file is already playable. Registering it would be an outright
    regression, not extra safety."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    monkeypatch.setattr("src.constants.METADATAPATH", "", raising=False)
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")
    registry = _recordingWriterRegistry(monkeypatch, ffmpegSettings)

    wb = ffmpegSettings.WriteBuffer(output=str(tmp_path / "out.mp4"))
    wb.writeBuffer.put(None)
    wb()  # returns on the sentinel, before any ffmpeg subprocess is spawned

    assert registry.added == []
    assert ffmpegSettings.finalizeLiveWriters(timeout=0.25) is True

    # Contrast, so the emptiness above means something: the Nelux writer does
    # register itself for the run of its loop.
    nwb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out2.mp4"), benchmark=True
    )
    nwb.writeBuffer.put(None)
    nwb()
    assert registry.added == ["NeluxWriteBuffer"]


def _saveProbe(monkeypatch, tmp_path, probe):
    """Run getVideoMetadata.saveMetadata against a tmp install dir.

    WHEREAMIRUNFROM and the METADATAPATH saveMetadata sets from it both go
    through monkeypatch so the session's real values come back afterwards.
    """
    import src.constants as cs

    getVideoMetadata = importlib.import_module("src.io.getVideoMetadata")
    monkeypatch.setattr(cs, "WHEREAMIRUNFROM", str(tmp_path))
    monkeypatch.setattr(cs, "METADATAPATH", "", raising=False)
    getVideoMetadata.saveMetadata(probe)
    return cs


def testSaveMetadataFeedsTheColourDecisionInProcess(monkeypatch, tmp_path):
    """The writers' colour decision now comes from cs.PROBED_METADATA, set by
    saveMetadata before any writer is built, instead of a file round-trip."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    probe = {"ColorSpace": "bt2020nc", "ColorTRT": "arib-std-b67"}
    cs = _saveProbe(monkeypatch, tmp_path, probe)

    assert cs.PROBED_METADATA == probe
    assert ffmpegSettings._probedMetadata() == probe


def testAConcurrentRunOverwritingMetadataJsonCannotFlipThisRunsColourSpace(
    monkeypatch, tmp_path
):
    """The race: metadata.json is one fixed install-dir path identical for every
    TAS process on the machine, so a second run starting mid-render overwrote
    this run's probe between saveMetadata and the writer's read -- a BT.2020
    source got converted and tagged bt709 (or the reverse), silently."""
    import json

    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    _saveProbe(
        monkeypatch,
        tmp_path,
        {"ColorSpace": "bt2020nc", "ColorTRT": "arib-std-b67"},
    )

    # The other run lands on the shared path with its own bt709 source.
    (tmp_path / "metadata.json").write_text(
        json.dumps({"metadata": {"ColorSpace": "bt709", "ColorTRT": "bt709"}})
    )

    assert ffmpegSettings._probedMetadata()["ColorSpace"] == "bt2020nc"
    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        encode_method="x265_nelux",
    )
    options = wb.encoderKwargs["options"]
    assert options["colorspace"] == "bt2020nc"
    assert options["color_primaries"] == "bt2020"
    assert options["color_trc"] == "arib-std-b67"


def testSaveMetadataStoresACopyOfTheProbeDict(monkeypatch, tmp_path):
    """getVideoMetadata hands saveMetadata the same dict it returns to main.py
    and callers keep using it, so aliasing would let a later mutation rewrite
    the colour decision the writers already committed to."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    probe = {"ColorSpace": "bt2020nc"}
    cs = _saveProbe(monkeypatch, tmp_path, probe)

    probe["ColorSpace"] = "bt709"
    probe["Width"] = 1920

    assert cs.PROBED_METADATA == {"ColorSpace": "bt2020nc"}
    assert ffmpegSettings._probedMetadata() == {"ColorSpace": "bt2020nc"}


def testProbedMetadataStillFallsBackToTheMetadataFile(monkeypatch, tmp_path):
    """Callers that point cs.METADATAPATH at a file without ever going through
    saveMetadata (the bt2020 test above, and tests/test_ffmpegColorspace.py)
    must keep working -- hence the "unset" sentinel stays None: a {} would be
    returned as a real answer and kill the file read silently."""
    import json

    import src.constants as cs

    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    metadataPath = tmp_path / "metadata.json"
    metadataPath.write_text(
        json.dumps({"metadata": {"ColorSpace": "bt2020nc", "ColorTRT": "smpte2084"}})
    )
    monkeypatch.setattr(cs, "METADATAPATH", str(metadataPath), raising=False)

    assert cs.PROBED_METADATA is None
    assert ffmpegSettings._probedMetadata() == {
        "ColorSpace": "bt2020nc",
        "ColorTRT": "smpte2084",
    }

    # An empty probe recorded in-process is an answer ("nothing to tag"), not
    # "unset", so the stale file must not be consulted behind it.
    monkeypatch.setattr(cs, "PROBED_METADATA", {})
    assert ffmpegSettings._probedMetadata() == {}
