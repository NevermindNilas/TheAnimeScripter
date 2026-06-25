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
from src.cli.parser import (
    DidYouMeanArgumentParser,
    TASHelpFormatter,
    _buildParser,
    _listMethods,
    capabilityMethods,
    str2bool,
)
from src.cli.sources import providedCliOptions, wasProvided
from src.cli.validator import (
    _autoEnableParentFlags,
    _configureProcessingSettings,
    isAnyOtherProcessingMethodEnabled,
)
from src.infra.backendFallback import applyBackendFallbacks, fallbackMethod


def makeArgs(**overrides):
    base = dict(
        slowmo=False,
        static_step=False,
        interpolate_factor=2.0,
        dedup=False,
        smooth_dedup=False,
        dedup_method="ssim",
        dedup_sens=35.0,
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


def testPysceneDetectSensInvertedOn0To100Scale():
    # AdaptiveDetector: higher threshold = fewer cuts, so user sens is flipped.
    a = makeArgs(autoclip=True, autoclip_method="pyscenedetect", autoclip_sens=30.0)
    _configureProcessingSettings(a)
    assert a.autoclip_sens == pytest.approx(70.0)


def testProbabilityBasedAutoclipSensMappedToUnitThreshold():
    # transnetv2 / maxxvit: sens 0..100 -> threshold 1..0 (higher sens = more cuts).
    a = makeArgs(autoclip=True, autoclip_method="transnetv2", autoclip_sens=30.0)
    _configureProcessingSettings(a)
    assert a.autoclip_sens == pytest.approx(0.70)


def testDedupDisablesAudio(monkeypatch):
    monkeypatch.setattr(cs, "AUDIO", True, raising=False)
    a = makeArgs(dedup=True, smooth_dedup=False, dedup_method="mse")
    _configureProcessingSettings(a)
    assert cs.AUDIO is False


def testSmoothDedupKeepsAudio(monkeypatch):
    monkeypatch.setattr(cs, "AUDIO", True, raising=False)
    a = makeArgs(dedup=True, smooth_dedup=True, dedup_method="mse")
    _configureProcessingSettings(a)
    assert cs.AUDIO is True


# --------------------------------------------------------------------------- #
# CLI source tracking and backend fallback
# --------------------------------------------------------------------------- #


def testProvidedCliOptionsNormalizesLongFlags():
    assert providedCliOptions(["--upscale-method=span", "--interpolate"]) == {
        "upscale_method",
        "interpolate",
    }


def testWasProvidedIncludesJsonKeys():
    args = types.SimpleNamespace(_json_keys={"interpolate_method"})
    assert wasProvided(args, "interpolate_method", set()) is True


def testAutoEnableParentFlagsUsesProvidedOptions():
    args = fullFlags()
    args.interpolate_method = "rife4.6"
    _autoEnableParentFlags(args, {"interpolate_method"})
    assert args.interpolate is True


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
