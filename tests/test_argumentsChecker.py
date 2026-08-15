"""Tests for the pure CLI helper logic.

Covers the parts that carry real logic rather than argparse plumbing:
- sensitivity remapping in _configureProcessingSettings (dedup/autoclip),
  where user-facing 0-100 scales get inverted/rescaled per method. This is the
  area recent fixes touched, so the exact transforms are pinned here.
- "did you mean?" fuzzy suggestion machinery.
- backend grouping for help output.
"""

import types

import pytest

import src.constants as cs
from src.cli.config import CliConfig
from src.cli.parser import (
    DidYouMeanArgumentParser,
    TASHelpFormatter,
    _buildParser,
    _listMethods,
    capabilityMethods,
    str2bool,
)
from src.cli.validation import (
    CliValidationError,
    applyRuntimeValidation,
    normalizeUpscaleFactor,
    parseOutputScale,
    selectedUpscaleBackend,
    validateCustomUpscaleModel,
)
from src.cli.validator import (
    _adjustMethodsBasedOnCuda,
    _applyImpliedFlags,
    _configureProcessingSettings,
    _downgradeCudaDetector,
    _handleDepthSettings,
    _mapAutoclipSensitivity,
    _mapDedupSensitivity,
    _resolveNeluxEncoder,
    isAnyOtherProcessingMethodEnabled,
)
from src.infra.backendFallback import applyBackendFallbacks, fallbackMethod


def makeArgs(**overrides):
    base = dict(
        slowmo=False,
        static_step=False,
        interpolate_factor=2.0,
        interpolate=False,
        interpolate_method="rife4.25",
        dedup=False,
        dedup_method="ssim",
        dedup_sens=35.0,
        smooth_dedup=False,
        smooth_dedup_method="ssim",
        smooth_dedup_sens=35.0,
        smooth_dedup_max_span=6,
        autoclip=False,
        autoclip_method="pyscenedetect",
        autoclip_sens=50.0,
        compile_mode="default",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# str2bool
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("val", ["yes", "true", "t", "y", "1", "TRUE"])
def testStr2boolTruthy(val):
    assert str2bool(val) is True


@pytest.mark.parametrize("val", ["no", "false", "f", "n", "0", "False"])
def testStr2boolFalsy(val):
    assert str2bool(val) is False


def testStr2boolPassesThroughBool():
    assert str2bool(True) is True


def testStr2boolRejectsGarbage():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")


# --------------------------------------------------------------------------- #
# isAnyOtherProcessingMethodEnabled
# --------------------------------------------------------------------------- #


def fullFlags(**on):
    flags = dict(
        interpolate=False,
        upscale=False,
        segment=False,
        restore=False,
        stabilize=False,
        resize=False,
        dedup=False,
        depth=False,
        autoclip=False,
        obj_detect=False,
        moblur=False,
    )
    flags.update(on)
    return types.SimpleNamespace(**flags)


def testNoProcessingEnabled():
    assert isAnyOtherProcessingMethodEnabled(fullFlags()) is False


def testSingleProcessingEnabled():
    assert isAnyOtherProcessingMethodEnabled(fullFlags(upscale=True)) is True


def testOutputScaleAloneCountsAsProcessing():
    """It used to count only because it auto-enabled --resize, which also
    applied --resize_factor's default of 2 to the decode."""
    assert (
        isAnyOtherProcessingMethodEnabled(fullFlags(output_scale="1920x1080")) is True
    )
    assert isAnyOtherProcessingMethodEnabled(fullFlags(output_scale="")) is False


# --------------------------------------------------------------------------- #
# *_nelux encoder resolution
# --------------------------------------------------------------------------- #


def neluxArgs(**overrides):
    base = dict(
        encode_method="x264_nelux",
        custom_encoder="",
        bit_depth="8bit",
        output_scale_width=None,
        output_scale_height=None,
        depth=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def testNeluxKeptWhenEveryOptionIsHonorable():
    args = neluxArgs()
    _resolveNeluxEncoder(args)
    assert args.encode_method == "x264_nelux"


def testNonNeluxMethodIsNeverRewritten():
    args = neluxArgs(encode_method="x264", bit_depth="16bit", depth=True)
    _resolveNeluxEncoder(args)
    assert args.encode_method == "x264"


@pytest.mark.parametrize(
    "overrides",
    [
        dict(custom_encoder="-c:v libx265 -crf 30"),
        dict(bit_depth="16bit"),
        dict(depth=True),
    ],
)
def testNeluxSwappedForItsTwinWhenAnOptionCannotBeHonored(overrides):
    """NeluxWriteBuffer takes these through **kwargs and never reads them, and
    --depth does not use it at all, so each used to be dropped in silence."""
    args = neluxArgs(**overrides)
    _resolveNeluxEncoder(args)
    assert args.encode_method == "x264"


def testOutputScaleKeepsNeluxWhenEncoderResizeIsSupported(monkeypatch):
    """nelux >= 0.18.0 scales encoder-side, so --output_scale no longer costs
    the in-process encoder."""
    import src.cli.validator as validator

    monkeypatch.setattr(validator, "_neluxSupportsEncoderResize", lambda: True)
    args = neluxArgs(output_scale_width=320, output_scale_height=180)
    _resolveNeluxEncoder(args)
    assert args.encode_method == "x264_nelux"


def testOutputScaleStillSwapsOnAPre018Nelux(monkeypatch):
    """An installed nelux older than 0.18.0 cannot resize encoder-side; keep
    the loud FFmpeg-twin downgrade instead of a silent drop or a raw TypeError
    from VideoEncoder(resize=...)."""
    import src.cli.validator as validator

    monkeypatch.setattr(validator, "_neluxSupportsEncoderResize", lambda: False)
    args = neluxArgs(output_scale_width=320, output_scale_height=180)
    _resolveNeluxEncoder(args)
    assert args.encode_method == "x264"


@pytest.mark.parametrize(
    "installed,expected",
    [
        ("0.17.0", False),
        ("0.18.0", True),
        ("0.18.1", True),
        ("1.0.0", True),
    ],
)
def testNeluxEncoderResizeVersionGate(monkeypatch, installed, expected):
    import src.cli.validator as validator

    monkeypatch.setattr("importlib.metadata.version", lambda name: installed)
    assert validator._neluxSupportsEncoderResize() is expected


def testNeluxEncoderResizeGateOpenWhenNeluxIsAbsent(monkeypatch):
    """No nelux installed (bare CI venv): the run dies at `import nelux` long
    before the writer matters, so don't rewrite the method on a failed probe."""
    from importlib.metadata import PackageNotFoundError

    import src.cli.validator as validator

    def raiser(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", raiser)
    assert validator._neluxSupportsEncoderResize() is True


def testNeluxEncoderResizeGateClosedOnUnparseableVersion(monkeypatch):
    """A version string the gate cannot parse downgrades loudly instead of
    letting a pre-0.18 build raise a raw TypeError at VideoEncoder(resize=...)."""
    import src.cli.validator as validator

    monkeypatch.setattr("importlib.metadata.version", lambda name: "unknowable")
    assert validator._neluxSupportsEncoderResize() is False


def _encodeMethodChoices():
    """The parser's --encode_method choices, straight from the built parser."""
    parser = _buildParser(".")
    for action in parser._actions:
        if "--encode_method" in action.option_strings:
            return list(action.choices)
    raise AssertionError("--encode_method not found in parser")


def testEveryNeluxMethodMapsToARealEncodeChoice():
    """Derived from the parser so a new *_nelux choice cannot ship without its
    FFmpeg twin: the twin must itself be a CLI choice with a matchEncoder arm,
    a matchNeluxEncoder mapping must exist, and that mapping must target the
    same codec the twin's FFmpeg arm names."""
    from src.io.encodingSettings import matchEncoder, matchNeluxEncoder

    choices = _encodeMethodChoices()
    neluxMethods = [m for m in choices if m.endswith("_nelux")]
    assert neluxMethods, "parser lost its *_nelux choices"

    for method in neluxMethods:
        args = neluxArgs(encode_method=method, depth=True)
        _resolveNeluxEncoder(args)
        twin = args.encode_method
        assert twin == method[: -len("_nelux")]
        assert twin in choices, f"{method} downgrades to non-choice {twin}"
        twinArgs = matchEncoder(twin)
        assert twinArgs, f"{twin} has no matchEncoder arm"

        mapping = matchNeluxEncoder(method)
        assert mapping is not None, f"{method} has no Nelux encoder mapping"
        assert mapping["codec"] in twinArgs, (
            f"{method} encodes {mapping['codec']} but its twin {twin} runs {twinArgs}"
        )


# --------------------------------------------------------------------------- #
# CUDA-only frame comparators
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "method,expected",
    [
        ("flownets", "ssim"),
        ("vmaf-cuda", "vmaf"),
        ("ssim-cuda", "ssim"),
        ("mse-cuda", "mse"),
        ("ssim", "ssim"),
        ("mse", "mse"),
    ],
)
def testCudaOnlyDetectorsDowngradeWithoutCuda(monkeypatch, method, expected):
    """--dedup used to skip this guard that --smooth_dedup had, so
    `--dedup --dedup_method flownets` on a Mac downloaded the weights and then
    died at model init with an error that never named the flag."""

    class _NoCuda:
        cudaAvailable = False

    monkeypatch.setattr("src.infra.isCudaInit.CudaChecker", _NoCuda)
    assert _downgradeCudaDetector(method, "dedup_method") == expected


@pytest.mark.parametrize("method", ["flownets", "vmaf-cuda", "ssim-cuda", "mse-cuda"])
def testCudaOnlyDetectorsSurviveWithCuda(monkeypatch, method):
    class _HasCuda:
        cudaAvailable = True

    monkeypatch.setattr("src.infra.isCudaInit.CudaChecker", _HasCuda)
    assert _downgradeCudaDetector(method, "dedup_method") == method


def fallbackArgs(**overrides):
    base = dict(
        supportsCuda=False,
        interpolate=False,
        interpolate_method="rife4.25",
        upscale=False,
        upscale_method="shufflecugan",
        segment=False,
        segment_method="anime",
        depth=False,
        depth_method="small_v2",
        restore=False,
        restore_method=["anime1080fixer"],
        dedup=False,
        dedup_method="ssim",
        obj_detect=False,
        obj_detect_method="yolov9_small-directml",
        moblur=False,
        moblur_method="rife4.25",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def testDarwinFallbackUsesMpsForMoblur(monkeypatch):
    monkeypatch.setattr(cs, "SYSTEM", "Darwin", raising=False)
    args = fallbackArgs(moblur=True, moblur_method="rife4.25")

    _adjustMethodsBasedOnCuda(args)

    assert args.moblur_method == "rife4.25-mps"


def testDarwinFallbackRewritesMoblurTensorRtToMps(monkeypatch):
    monkeypatch.setattr(cs, "SYSTEM", "Darwin", raising=False)
    args = fallbackArgs(moblur=True, moblur_method="rife4.6-tensorrt")

    _adjustMethodsBasedOnCuda(args)

    assert args.moblur_method == "rife4.6-mps"


def testMoblurParserIncludesMpsChoices():
    methods = capabilityMethods(_buildParser("."))["moblur"]
    assert "rife4.6-mps" in methods
    assert "rife4.25-mps" in methods


def testCudalessFallbackRewritesSegmentToDirectml(monkeypatch):
    # --segment with its DEFAULT method used to survive the fallback untouched
    # and then construct AnimeSegment, whose unguarded torch.cuda.Stream()
    # raised on every machine without CUDA. The fallback looked the method up
    # in modelsList(), which registers segment as "segment*" while the CLI
    # calls it "anime*", so "anime-directml" was never found.
    monkeypatch.setattr(cs, "SYSTEM", "Windows", raising=False)
    args = fallbackArgs(segment=True, segment_method="anime")

    _adjustMethodsBasedOnCuda(args)

    assert args.segment_method == "anime-directml"


def testCudalessFallbackRewritesRife425(monkeypatch):
    # modelsList() carries no rife*-directml entries at all, so every 4.25-family
    # method also survived unchanged into RifeCuda's unguarded CUDA streams.
    monkeypatch.setattr(cs, "SYSTEM", "Windows", raising=False)
    args = fallbackArgs(interpolate=True, interpolate_method="rife4.25")

    _adjustMethodsBasedOnCuda(args)

    assert args.interpolate_method == "rife4.25-directml"


def testCudalessFallbackKeepsRifeOnNcnn(monkeypatch):
    # RIFE's NCNN path runs about an order of magnitude faster than its
    # DirectML one, so interpolation must not be handed to DirectML merely
    # because a -directml choice exists. Switching the fallback to the CLI
    # choice list made both spellings visible for the first time, and the
    # default --interpolate_method silently flipped ncnn -> directml.
    monkeypatch.setattr(cs, "SYSTEM", "Windows", raising=False)
    args = fallbackArgs(interpolate=True, interpolate_method="rife4.6")

    _adjustMethodsBasedOnCuda(args)

    assert args.interpolate_method == "rife4.6-ncnn"


def testFallbackPrefersDirectmlOutsideInterpolation():
    # The ncnn-first preference is interpolation-specific.
    choices = ["shufflecugan-directml", "shufflecugan-ncnn"]
    assert (
        fallbackMethod("shufflecugan", set(), choices=choices, capability="upscale")
        == "shufflecugan-directml"
    )
    assert (
        fallbackMethod(
            "rife4.6",
            set(),
            choices=["rife4.6-directml", "rife4.6-ncnn"],
            capability="interpolate",
        )
        == "rife4.6-ncnn"
    )


def testFallbackMethodPrefersCliChoicesOverRegistry():
    # The registry name and the CLI name disagree for segment; the CLI name wins.
    assert fallbackMethod("anime", {"segment-directml"}) == "anime"
    assert (
        fallbackMethod("anime", {"segment-directml"}, choices=["anime-directml"])
        == "anime-directml"
    )


def testDarwinFallbackUsesMpsForDepth(monkeypatch):
    monkeypatch.setattr(cs, "SYSTEM", "Darwin", raising=False)
    args = fallbackArgs(depth=True, depth_method="small_v2")

    _adjustMethodsBasedOnCuda(args)

    assert args.depth_method == "small_v2-mps"


def testDepthParserIncludesMpsChoices():
    methods = capabilityMethods(_buildParser("."))["depth"]
    assert "small_v2-mps" in methods
    assert "large_v3-mps" in methods


def testDepthBatchRetainedOnMps():
    args = types.SimpleNamespace(
        depth=True,
        depth_quality="low",
        depth_method="small_v2-mps",
        depth_norm=False,
        depth_batch=4,
        half=False,
    )

    _handleDepthSettings(args)

    assert args.depth_batch == 4


# --------------------------------------------------------------------------- #
# _configureProcessingSettings: sensitivity remapping
# --------------------------------------------------------------------------- #


def testSsimDedupSensRemapped():
    a = makeArgs(dedup=True, dedup_method="ssim", dedup_sens=35.0)
    _configureProcessingSettings(a)
    assert a.dedup_sens == pytest.approx(1.0 - 35.0 / 1000)  # 0.965


def testFlownetsDedupSensDividedBy100():
    a = makeArgs(dedup=True, dedup_method="flownets", dedup_sens=20.0)
    _configureProcessingSettings(a)
    assert a.dedup_sens == pytest.approx(0.20)


def testMseDedupSensUntouched():
    a = makeArgs(dedup=True, dedup_method="mse", dedup_sens=20.0)
    _configureProcessingSettings(a)
    assert a.dedup_sens == 20.0


def testPysceneDetectSensMappedOntoAdaptiveThresholdScale():
    # AdaptiveDetector: higher threshold = fewer cuts, so user sens is flipped —
    # onto the detector's own ~0.5-6 ratio scale, where 50 is the library
    # default of 3.0. (The old `100 - sens` mapping made the default 50, which
    # detected nothing.)
    a = makeArgs(autoclip=True, autoclip_method="pyscenedetect", autoclip_sens=30.0)
    _configureProcessingSettings(a)
    assert a.autoclip_sens == pytest.approx(4.2)


def testProbabilityBasedAutoclipSensMappedToUnitThreshold():
    # transnetv2 / maxxvit: sens 0..100 -> threshold 1..0 (higher sens = more cuts).
    a = makeArgs(autoclip=True, autoclip_method="transnetv2", autoclip_sens=30.0)
    _configureProcessingSettings(a)
    assert a.autoclip_sens == pytest.approx(0.70)


def testDedupDisablesAudio(monkeypatch):
    monkeypatch.setattr(cs, "AUDIO", True, raising=False)
    a = makeArgs(dedup=True, dedup_method="mse")
    _configureProcessingSettings(a)
    assert cs.AUDIO is False


# --------------------------------------------------------------------------- #
# _configureProcessingSettings: --smooth_dedup
# --------------------------------------------------------------------------- #


def testSmoothDedupKeepsAudio(monkeypatch):
    # --smooth_dedup preserves duration, so audio stays in sync and stays enabled.
    monkeypatch.setattr(cs, "AUDIO", True, raising=False)
    a = makeArgs(smooth_dedup=True, smooth_dedup_method="mse")
    _configureProcessingSettings(a)
    assert cs.AUDIO is True


def testSmoothDedupAutoEnablesInterpolate():
    a = makeArgs(smooth_dedup=True, interpolate=False)
    _applyImpliedFlags(a)
    assert a.interpolate is True


def testSmoothDedupDisablesDedup():
    # The frame-dropping path would shorten the video --smooth_dedup preserves.
    a = makeArgs(smooth_dedup=True, dedup=True, dedup_method="mse")
    _applyImpliedFlags(a)
    assert a.dedup is False


def testImpliedFlagsRunBeforeTheGuardsThatReadThem():
    """--slowmo and --dynamic_scale are gated on --interpolate by guards that run
    before _configureProcessingSettings, so --smooth_dedup has to have enabled it
    by then or both flags switch themselves off with a misleading message."""
    a = makeArgs(smooth_dedup=True, interpolate=False, dynamic_scale=True)
    _applyImpliedFlags(a)
    assert a.interpolate is True

    # Same call order prepareRuntimeArgs uses.
    assert not (a.slowmo and not a.interpolate)
    _configureProcessingSettings(a)
    assert a.dynamic_scale is True


@pytest.mark.parametrize(
    "method, expected",
    [
        ("ssim", 1.0 - 35.0 / 1000),
        ("ssim-cuda", 1.0 - 35.0 / 1000),
        ("vmaf", 100.0 - 3.5),
        ("vmaf-cuda", 100.0 - 3.5),
        ("flownets", 0.35),
        ("mse", 35.0),
        ("mse-cuda", 35.0),
    ],
)
def testSmoothDedupSensUsesTheSameMappingAsDedup(method, expected):
    # The shared helper is what both --dedup and --smooth_dedup call, so testing
    # it directly keeps the table independent of the CUDA-availability downgrade.
    assert _mapDedupSensitivity(method, 35.0) == pytest.approx(expected)


def testSmoothDedupSensIsMappedOnTheArgs():
    a = makeArgs(smooth_dedup=True, smooth_dedup_method="mse", smooth_dedup_sens=20.0)
    _configureProcessingSettings(a)
    assert a.smooth_dedup_sens == 20.0


@pytest.mark.parametrize(
    "method, sens, expected",
    [
        # pyscenedetect: 50 must land on AdaptiveDetector's library default of
        # 3.0. The old `100 - sens` mapping made the default threshold 50 -- a
        # frame-score ratio virtually never reached, so a default run detected
        # nothing, not even an artificial hard cut.
        ("pyscenedetect", 50.0, 3.0),
        ("pyscenedetect", 0.0, 6.0),
        ("pyscenedetect", 100.0, 0.5),  # floored: 0 would cut on every frame
        ("pyscenedetect", 97.0, 0.5),  # 0.18 -> floor
        # probability-threshold backends keep the inverted-fraction mapping
        ("maxxvit-tensorrt", 50.0, 0.5),
        ("maxxvit-directml", 30.0, 0.7),
        ("transnetv2", 50.0, 0.5),
    ],
)
def testAutoclipSensMapsOntoBackendThreshold(method, sens, expected):
    assert _mapAutoclipSensitivity(method, sens) == pytest.approx(expected)


def testSmoothDedupMaxSpanZeroMeansUncappedAndNegativesNormalize():
    # 0 disables the cap (main.py's truthiness guard skips the span check);
    # negatives must normalize to 0, not survive to make `span > cap`
    # always-true and hold every gap.
    for given, expected in ((-3, 0), (0, 0), (1, 1), (6, 6)):
        a = makeArgs(smooth_dedup=True, smooth_dedup_max_span=given)
        _configureProcessingSettings(a)
        assert a.smooth_dedup_max_span == expected


# --------------------------------------------------------------------------- #
# CLI source tracking and backend fallback
# --------------------------------------------------------------------------- #


def testProvidedCliOptionsNormalizesLongFlags():
    assert CliConfig.collectProvidedOptions(
        ["--upscale-method=span", "--interpolate"]
    ) == {
        "upscale_method",
        "interpolate",
    }


def testWasProvidedIncludesJsonKeys():
    config = CliConfig(
        args=None,
        parser=None,
        argv=[],
        providedOptions=set(),
        jsonKeys={"interpolate_method"},
    )
    assert config.optionWasProvided("interpolate_method")


def testAutoEnableParentFlagsUsesProvidedOptions():
    args = fullFlags()
    args.interpolate_method = "rife4.6"
    config = CliConfig(
        args=args,
        parser=None,
        argv=[],
        providedOptions={"interpolate_method"},
        jsonKeys=set(),
    )
    config.autoEnableParentFlags()
    assert args.interpolate is True


def testNormalizeCliConfigLoadsJsonAndAutoEnablesParent(tmp_path, builtParser):
    configPath = tmp_path / "tas.json"
    configPath.write_text('{"upscale_method": "span"}', encoding="utf-8")
    args = builtParser.parse_args(["--json", str(configPath)])

    cliConfig = CliConfig.fromArgs(args, builtParser, ["--json", str(configPath)])

    assert cliConfig.jsonKeys == {"upscale_method"}
    assert args.upscale_method == "span"
    assert args.upscale is True


def testNormalizeCliConfigRejectsJsonMixedWithOtherOptions(tmp_path, builtParser):
    configPath = tmp_path / "tas.json"
    configPath.write_text("{}", encoding="utf-8")
    args = builtParser.parse_args(["--json", str(configPath), "--upscale"])

    with pytest.raises(SystemExit):
        CliConfig.fromArgs(
            args,
            builtParser,
            ["--json", str(configPath), "--upscale"],
        )


def testFallbackMethodPrefersMpsWhenAvailable():
    models = {"rife4.6-mps", "rife4.6-directml", "rife4.6-ncnn"}
    assert fallbackMethod("rife4.6", models, preferMps=True) == "rife4.6-mps"


def testFallbackMethodUsesDirectmlBeforeNcnn():
    models = {"rife4.6-directml", "rife4.6-ncnn"}
    assert fallbackMethod("rife4.6", models) == "rife4.6-directml"


def testApplyBackendFallbackSkipsExplicitBackend():
    args = fullFlags(upscale=True)
    args.upscale_method = "span-tensorrt"
    applyBackendFallbacks(args, {"span-directml"})
    assert args.upscale_method == "span-tensorrt"


def testApplyBackendFallbackHandlesRestoreLists():
    args = fullFlags(restore=True)
    args.restore_method = ["anime1080fixer", "scunet-directml"]
    applyBackendFallbacks(args, {"anime1080fixer-ncnn", "scunet-ncnn"})
    assert args.restore_method == ["anime1080fixer-ncnn", "scunet-directml"]


# --------------------------------------------------------------------------- #
# CLI validation helpers
# --------------------------------------------------------------------------- #


def testParseOutputScaleAcceptsWidthByHeight():
    assert parseOutputScale("2560x1440") == (2560, 1440)


@pytest.mark.parametrize("value", ["bad", "0x1080", "1920x0", "1920xabc"])
def testParseOutputScaleRejectsInvalidValues(value):
    with pytest.raises(CliValidationError):
        parseOutputScale(value)


def testNormalizeUpscaleFactorClampsTooSmall():
    args = types.SimpleNamespace(upscale=True, upscale_factor=1)
    warning = normalizeUpscaleFactor(args)
    assert args.upscale_factor == 2
    assert "at least 2" in warning


def testNormalizeUpscaleFactorClampsInvalidValue():
    args = types.SimpleNamespace(upscale=True, upscale_factor="abc")
    warning = normalizeUpscaleFactor(args)
    assert args.upscale_factor == 2
    assert "Invalid upscale_factor" in warning


def testSelectedUpscaleBackendSplitsBackendSuffix():
    assert selectedUpscaleBackend("span-directml") == ("directml", "span")
    assert selectedUpscaleBackend("span") == ("pytorch", "span")


def testValidateCustomUpscaleModelAcceptsOnnxBackend(tmp_path):
    model = tmp_path / "custom.onnx"
    model.write_bytes(b"model")
    args = types.SimpleNamespace(
        custom_model=str(model),
        upscale_method="span-directml",
    )
    validateCustomUpscaleModel(args)
    assert args.custom_model == str(model.resolve())


def testValidateCustomUpscaleModelRejectsOnnxWithoutOnnxBackend(tmp_path):
    model = tmp_path / "custom.onnx"
    model.write_bytes(b"model")
    args = types.SimpleNamespace(custom_model=str(model), upscale_method="span")
    with pytest.raises(CliValidationError):
        validateCustomUpscaleModel(args)


def testValidateCustomUpscaleModelRejectsPytorchWithOnnxBackend(tmp_path):
    model = tmp_path / "custom.pth"
    model.write_bytes(b"model")
    args = types.SimpleNamespace(
        custom_model=str(model),
        upscale_method="span-directml",
    )
    with pytest.raises(CliValidationError):
        validateCustomUpscaleModel(args)


@pytest.mark.parametrize("outpoint", [5.0, 4.0])
def testRuntimeValidationRejectsOutpointNotAfterInpoint(outpoint):
    args = types.SimpleNamespace(
        custom_model=None,
        output_scale=None,
        upscale=False,
        inpoint=5.0,
        outpoint=outpoint,
    )

    with pytest.raises(CliValidationError, match="outpoint must be greater"):
        applyRuntimeValidation(args)


def testRuntimeValidationAllowsZeroOutpoint():
    args = types.SimpleNamespace(
        custom_model=None,
        output_scale=None,
        upscale=False,
        inpoint=5.0,
        outpoint=0.0,
    )

    applyRuntimeValidation(args)


# --------------------------------------------------------------------------- #
# DidYouMeanArgumentParser: fuzzy suggestions
# --------------------------------------------------------------------------- #


@pytest.fixture
def parser():
    return DidYouMeanArgumentParser()


def testExactMatchScoresHighest(parser):
    choices = ["rife4.6", "rife4.22", "scunet"]
    best = parser.getSuggestions("rife4.6", choices)
    assert best[0] == "rife4.6"


def testSuggestionsFilterOutUnrelated(parser):
    # A wildly different choice falls below the 0.3 threshold and is dropped.
    suggestions = parser.getSuggestions("rife4.6", ["rife4.6", "x264"])
    assert "x264" not in suggestions


def testSuggestionsRespectMax(parser):
    choices = [f"rife4.{i}" for i in range(20)]
    assert len(parser.getSuggestions("rife4.1", choices, maxSuggestions=3)) <= 3


# --------------------------------------------------------------------------- #
# TASHelpFormatter._group_choices: split method list by backend suffix
# --------------------------------------------------------------------------- #


def testGroupChoicesBucketsByBackend():
    groups = TASHelpFormatter._group_choices(
        ["span", "span-tensorrt", "span-ncnn", "rtmosr-directml"]
    )
    assert groups["cuda"] == ["span"]
    assert groups["tensorrt"] == ["span-tensorrt"]
    assert groups["ncnn"] == ["span-ncnn"]
    assert groups["directml"] == ["rtmosr-directml"]


def testGroupChoicesUnknownSuffixFallsToCuda():
    # A trailing token that isn't a known backend stays in the default cuda bucket.
    groups = TASHelpFormatter._group_choices(["rife4.25-heavy"])
    assert groups["cuda"] == ["rife4.25-heavy"]


# --------------------------------------------------------------------------- #
# Family-aware scoring + invalid-choice error output
# --------------------------------------------------------------------------- #


def testFamilyBonusRanksSameBaseFirst(parser):
    # 'span-trt' shares the base 'span', so span-* should outrank unrelated names.
    choices = ["span-tensorrt", "span-ncnn", "x-cuda", "shufflecugan-tensorrt"]
    assert parser.getSuggestions("span-trt", choices)[0] == "span-tensorrt"


def testInvalidChoiceSuggestionNotDoubleQuoted(parser, capsys):
    # Regression: argparse wraps choices in quotes; we must strip before repr()
    # so the output is 'span-tensorrt', never "'span-tensorrt'".
    parser.add_argument("--upscale_method", choices=["span-tensorrt", "x-cuda"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--upscale_method", "span-tensorrtt"])
    err = capsys.readouterr().err
    assert "Did you mean" in err
    assert "'span-tensorrt'" in err
    assert "\"'" not in err


# --------------------------------------------------------------------------- #
# Misspelled option-name suggestions (new feature)
# --------------------------------------------------------------------------- #


def testSuggestOptionsClosestLongFlag(parser):
    parser.add_argument("--upscale", action="store_true")
    parser.add_argument("--upscale_method")
    parser.add_argument("--interpolate", action="store_true")
    opts = parser._collectOptionStrings()
    assert parser._suggestOptions("--upscalee", opts)[0] == "--upscale"


def testSuggestOptionsRespectsDashStyle(parser):
    # A short-style typo must never be matched against long options.
    parser.add_argument("--input")
    opts = parser._collectOptionStrings()
    assert all(not o.startswith("--") for o in parser._suggestOptions("-i", opts))


def testUnrecognizedOptionPrintsSuggestion(parser, capsys):
    parser.add_argument("--upscale", action="store_true")
    with pytest.raises(SystemExit):
        parser.parse_args(["--upscalee"])
    err = capsys.readouterr().err
    assert "Did you mean" in err
    assert "--upscale" in err


def testStrayPositionalFallsThrough(parser, capsys):
    # Non-flag unrecognized tokens are not options -> default argparse handling.
    parser.add_argument("--upscale", action="store_true")
    with pytest.raises(SystemExit):
        parser.parse_args(["somefile.mp4"])
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err
    assert "Did you mean" not in err


def testGenericErrorUsesUnifiedStyle(parser, capsys):
    # Non-suggestion errors (bad value, missing arg, ...) use the same minimal
    # style: "Error:" prefix, no usage block, no "main.py: error:".
    parser.add_argument("--inpoint", type=float)
    with pytest.raises(SystemExit):
        parser.parse_args(["--inpoint", "abc"])
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "usage:" not in err
    assert "main.py: error:" not in err


# --------------------------------------------------------------------------- #
# capabilityMethods + --list_methods (single-source method registry / drift guard)
# --------------------------------------------------------------------------- #


@pytest.fixture
def builtParser():
    return _buildParser(".")


def testCapabilityMethodsExcludesDecode(builtParser):
    caps = capabilityMethods(builtParser)
    assert "decode" not in caps  # decode_method (cpu/nvdec) is a backend toggle
    for expected in ("upscale", "interpolate", "restore", "depth", "obj_detect"):
        assert expected in caps


def testCapabilityMethodsCoverAllMethodDests(builtParser):
    # Source of truth: every *_method action with choices (bar decode_method).
    methodDests = {
        a.dest
        for a in builtParser._actions
        if a.dest.endswith("_method") and a.choices and a.dest != "decode_method"
    }
    expected = {d[: -len("_method")] for d in methodDests}
    assert expected == set(capabilityMethods(builtParser))


def testNoDuplicateMethodChoices(builtParser):
    # 2E drift guard: the hand-maintained choice lists must not gain duplicates.
    dupes = {}
    for capability, methods in capabilityMethods(builtParser).items():
        repeated = sorted({m for m in methods if methods.count(m) > 1})
        if repeated:
            dupes[capability] = repeated
    assert dupes == {}, f"Duplicate method choices: {dupes}"


def testListMethodsUnknownReturns2WithSuggestion(builtParser, capsys):
    assert _listMethods(builtParser, "upscal") == 2
    assert "upscale" in capsys.readouterr().err


def testListMethodsAllReturns0(builtParser, capsys):
    assert _listMethods(builtParser, "all") == 0
    out = capsys.readouterr().out
    assert "upscale" in out and "interpolate" in out


def testBannerOnlyOnFullHelpNotUsage(builtParser):
    # The banner belongs on --help only; usage-only output (which argparse
    # reuses on every error via format_usage) must not carry it.
    assert "AI-powered" in builtParser.format_help()
    assert "AI-powered" not in builtParser.format_usage()


# --------------------------------------------------------------------------- #
# --json capability auto-enabling
# --------------------------------------------------------------------------- #


def _jsonConfig(tmp_path, monkeypatch, payload):
    import json as _json

    path = tmp_path / "cfg.json"
    path.write_text(_json.dumps(payload), encoding="utf-8")
    parser = _buildParser("output")
    args = parser.parse_args(["--json", str(path)])
    monkeypatch.setattr("sys.argv", ["main.py", "--json", str(path)])
    CliConfig.fromArgs(args, parser, ["--json", str(path)])
    return args


def testJsonMethodAtItsDefaultDoesNotEnableACapability(tmp_path, monkeypatch):
    """The After Effects panel serializes its whole form, so every capability's
    *_method arrives at its default value. Treating those as 'provided' turned
    on upscale, depth, segment, dedup and restore for a config that asked for
    interpolation alone -- the run produced a depth map."""
    args = _jsonConfig(
        tmp_path,
        monkeypatch,
        {
            "interpolate": True,
            "interpolate_factor": 2,
            "upscale_method": "shufflecugan",
            "depth_method": "small_v2",
            "segment_method": "anime",
            "dedup_method": "ssim",
        },
    )
    assert args.interpolate is True
    assert (args.upscale, args.depth, args.segment, args.dedup) == (
        False,
        False,
        False,
        False,
    )


def testJsonExplicitFalseBeatsASiblingMethodKey(tmp_path, monkeypatch):
    """`"upscale": false` alongside `"upscale_method"` used to upscale anyway."""
    args = _jsonConfig(
        tmp_path,
        monkeypatch,
        {"upscale": False, "upscale_method": "span", "interpolate": True},
    )
    assert args.upscale is False


def testJsonNonDefaultMethodStillEnablesItsCapability(tmp_path, monkeypatch):
    """A config that genuinely asks for a model, without naming the flag, keeps
    working -- the whole point of the auto-enable. Reaches the parent flag
    through the `currentValue != defaultValue` fallback rather than jsonKeys,
    so it holds regardless of which of the two paths fires."""
    args = _jsonConfig(tmp_path, monkeypatch, {"upscale_method": "span"})
    assert args.upscale is True
    assert args.upscale_method == "span"


def testJsonExplicitTrueIsNotUndoneByTheNewGuard(tmp_path, monkeypatch):
    """The guard skips auto-enabling only when the config says the capability
    is OFF; an explicit true must still run."""
    args = _jsonConfig(
        tmp_path, monkeypatch, {"upscale": True, "upscale_method": "shufflecugan"}
    )
    assert args.upscale is True
