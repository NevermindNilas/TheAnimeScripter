"""Tests for src/utils/argumentsChecker.py — pure CLI helper logic.

Covers the parts that carry real logic rather than argparse plumbing:
- sensitivity remapping in _configureProcessingSettings (dedup/sharpen/autoclip),
  where user-facing 0-100 scales get inverted/rescaled per method. This is the
  area recent fixes touched, so the exact transforms are pinned here.
- "did you mean?" fuzzy suggestion machinery.
- backend grouping for help output.
"""

import types

import pytest

import src.constants as cs
from src.utils.argumentsChecker import (
    str2bool,
    isAnyOtherProcessingMethodEnabled,
    _configureProcessingSettings,
    DidYouMeanArgumentParser,
    TASHelpFormatter,
)


def makeArgs(**overrides):
    base = dict(
        slowmo=False, static_step=False, interpolate_factor=2.0,
        dedup=False, smooth_dedup=False, dedup_method="ssim", dedup_sens=35.0,
        sharpen=False, sharpen_sens=50.0,
        autoclip=False, autoclip_method="pyscenedetect", autoclip_sens=50.0,
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
        interpolate=False, upscale=False, segment=False, restore=False,
        stabilize=False, sharpen=False, resize=False, dedup=False, depth=False,
        autoclip=False, obj_detect=False, moblur=False,
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


def testSharpenSensDividedBy100():
    a = makeArgs(sharpen=True, sharpen_sens=50.0)
    _configureProcessingSettings(a)
    assert a.sharpen_sens == pytest.approx(0.5)


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
# DidYouMeanArgumentParser: fuzzy suggestions
# --------------------------------------------------------------------------- #

@pytest.fixture
def parser():
    return DidYouMeanArgumentParser()


def testLevenshteinKnownDistance(parser):
    assert parser._levenshteinDistance("kitten", "sitting") == 3


def testLevenshteinSymmetric(parser):
    assert parser._levenshteinDistance("abc", "abcd") == parser._levenshteinDistance("abcd", "abc")


def testLevenshteinIdenticalIsZero(parser):
    assert parser._levenshteinDistance("rife", "rife") == 0


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
