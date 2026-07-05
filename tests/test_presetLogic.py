"""Tests for src.server.presetLogic — --preset save/load round-trip.

createPreset is dual-purpose: an unknown preset name SAVES the current args
(minus the per-run I/O fields), a known name LOADS the stored values onto the
args namespace. The ignoreList is the contract that matters — if input/output
ever leak into a preset, every later run with that preset would overwrite the
same file. Corrupt presets.json must be survivable: presets are convenience
state, not something that may abort a render.
"""

import json
import os
import types

import pytest

import src.constants as cs
from src.server.presetLogic import createPreset, listPresets


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
        interpolate=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _stored(presetDir):
    with open(os.path.join(presetDir, "presets.json")) as f:
        return json.load(f)["Presets"]


def testNewPresetIsSaved(presetDir):
    createPreset(makeArgs(upscale=True, upscale_factor=4))
    stored = _stored(presetDir)
    assert stored["myPreset"]["upscale"] is True
    assert stored["myPreset"]["upscale_factor"] == 4


def testPerRunIOFieldsAreNeverStored(presetDir):
    createPreset(makeArgs())
    stored = _stored(presetDir)["myPreset"]
    for key in ("input", "output", "inpoint", "outpoint"):
        assert key not in stored


def testExistingPresetLoadsOntoArgs(presetDir):
    createPreset(makeArgs(upscale=True, upscale_factor=4))
    # Fresh invocation with the same preset name and different settings: the
    # stored values must win.
    args = createPreset(makeArgs(upscale=False, upscale_factor=2))
    assert args.upscale is True
    assert args.upscale_factor == 4


def testLoadingPresetKeepsPerRunIOFields(presetDir):
    createPreset(makeArgs())
    args = createPreset(makeArgs(input="other.mp4", output="elsewhere.mp4"))
    assert args.input == "other.mp4"
    assert args.output == "elsewhere.mp4"


def testLoadingDoesNotRewriteFile(presetDir):
    createPreset(makeArgs(upscale=True))
    before = _stored(presetDir)
    createPreset(makeArgs(upscale=False))
    assert _stored(presetDir) == before


def testTwoPresetsCoexist(presetDir):
    createPreset(makeArgs(preset="a", upscale=True))
    createPreset(makeArgs(preset="b", interpolate=True))
    stored = _stored(presetDir)
    assert stored["a"]["upscale"] is True and not stored["a"]["interpolate"]
    assert stored["b"]["interpolate"] is True and not stored["b"]["upscale"]


def testCorruptPresetsFileIsRecovered(presetDir):
    (presetDir / "presets.json").write_text("{not valid json", encoding="utf-8")
    createPreset(makeArgs(upscale=True))
    assert _stored(presetDir)["myPreset"]["upscale"] is True


def testListPresetsPrintsNamesAndEnabledFlags(presetDir, capsys):
    createPreset(makeArgs(preset="quality", upscale=True))
    capsys.readouterr()  # drop the "created successfully" line
    listPresets()
    out = capsys.readouterr().out
    assert "quality" in out
    assert "upscale" in out


def testListPresetsWithoutFile(presetDir, capsys):
    listPresets()
    assert "No presets found" in capsys.readouterr().out
