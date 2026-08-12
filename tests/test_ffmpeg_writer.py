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
