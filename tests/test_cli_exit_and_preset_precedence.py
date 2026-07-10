"""Regression tests for two CLI bugs.

BUG C: bare ``sys.exit()`` returns 0, so validation ERROR paths reported
success. The fix routes error exits through ``sys.exit(1)`` while leaving
informational/success short-circuits at 0. ``CliConfig.loadJsonConfig`` is the
most isolated ERROR site to exercise directly.

BUG D: a loaded preset stored all of ``vars(args)`` and clobbered every flag on
load, including ones the user explicitly typed. The fix makes an explicitly
provided CLI flag beat the preset (explicit CLI flag > preset > default) via a
``providedOptions`` guard, without breaking the save branch.
"""

import json
import os
import types

import pytest

import src.constants as cs
from src.cli.config import CliConfig
from src.server.presetLogic import createPreset


@pytest.fixture
def presetDir(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "WHEREAMIRUNFROM", str(tmp_path))
    return tmp_path


def makeArgs(**overrides):
    base = dict(
        preset="myPreset",
        input="in.mp4",
        output="out.mp4",
        inpoint=1.0,
        outpoint=2.0,
        upscale=False,
        upscale_factor=2,
        upscale_method="shufflecugan",
        encode_method="x264",
        interpolate=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _stored(presetDir):
    with open(os.path.join(presetDir, "presets.json")) as f:
        return json.load(f)["Presets"]


# --- BUG D -----------------------------------------------------------------


def testExplicitFlagBeatsLoadedPreset(presetDir):
    # Save a preset carrying span / x265.
    createPreset(makeArgs(preset="q", upscale_method="span", encode_method="x265"))
    # Fresh run reuses the preset but the user typed --upscale_method and
    # --encode_method explicitly; those must survive the load.
    args = createPreset(
        makeArgs(preset="q", upscale_method="shufflecugan", encode_method="x264"),
        providedOptions={"upscale_method", "encode_method"},
    )
    assert args.upscale_method == "shufflecugan"
    assert args.encode_method == "x264"


def testPresetStillFillsFlagsUserDidNotProvide(presetDir):
    createPreset(makeArgs(preset="q", upscale=True, upscale_factor=4))
    # Only upscale_method was typed on the CLI; the rest comes from the preset.
    args = createPreset(
        makeArgs(preset="q", upscale=False, upscale_factor=2, upscale_method="span"),
        providedOptions={"upscale_method"},
    )
    assert args.upscale is True
    assert args.upscale_factor == 4
    assert args.upscale_method == "span"


def testSavingNewPresetStillWorksWithProvidedOptions(presetDir):
    # The save branch must ignore providedOptions entirely.
    createPreset(
        makeArgs(preset="new", upscale=True, upscale_factor=4),
        providedOptions={"upscale", "upscale_factor"},
    )
    stored = _stored(presetDir)["new"]
    assert stored["upscale"] is True
    assert stored["upscale_factor"] == 4


def testSavingNewPresetWorksWithoutProvidedOptions(presetDir):
    # Back-compat: createPreset is still callable with a single argument.
    createPreset(makeArgs(preset="legacy", upscale=True))
    assert _stored(presetDir)["legacy"]["upscale"] is True


# --- BUG C -----------------------------------------------------------------


def testMissingJsonConfigExitsNonZero(tmp_path):
    args = types.SimpleNamespace(json=str(tmp_path / "does_not_exist.json"))
    config = CliConfig(
        args=args, parser=None, argv=[], providedOptions=set(), jsonKeys=set()
    )
    with pytest.raises(SystemExit) as excinfo:
        config.loadJsonConfig()
    assert excinfo.value.code == 1


def testInvalidJsonConfigExitsNonZero(tmp_path):
    badPath = tmp_path / "bad.json"
    badPath.write_text("{not valid json", encoding="utf-8")
    args = types.SimpleNamespace(json=str(badPath))
    config = CliConfig(
        args=args, parser=None, argv=[], providedOptions=set(), jsonKeys=set()
    )
    with pytest.raises(SystemExit) as excinfo:
        config.loadJsonConfig()
    assert excinfo.value.code == 1
