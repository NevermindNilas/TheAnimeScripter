import importlib
import importlib.util
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


NELUX_TWINS = [
    ("x264_nelux", "x264"),
    ("x265_nelux", "x265"),
    ("av1_nelux", "av1"),
    ("nvenc_h264_nelux", "nvenc_h264"),
    ("nvenc_h265_nelux", "nvenc_h265"),
    ("nvenc_av1_nelux", "nvenc_av1"),
]


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

    for method in ("x264", "slow_x265", "lossless_nvenc", "prores_segment", "png"):
        wb = ffmpegSettings.WriteBuffer(
            output=str(tmp_path / "out.mp4"), encode_method=method
        )
        assert wb.encode_method == method


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
