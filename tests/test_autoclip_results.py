"""Tests for src.autoclip._results — per-video cut-list writing.

A batch used to write every video's cuts to one fixed install-dir
autoclipresults.txt, so only the last video's results survived and --output
was ignored. The writer now targets the manufactured per-video output path;
in ADOBE mode a legacy copy is kept for the After Effects panel.
"""

import os

import src.constants as cs
from src.autoclip._results import writeCutResults


def testWritesCutsOnePerLine(tmp_path):
    out = tmp_path / "clip-Autoclip.txt"
    writeCutResults([1.5, 3.25], str(out))
    assert out.read_text() == "1.5\n3.25\n"


def testZeroCutsWritesEmptyFile(tmp_path):
    out = tmp_path / "clip-Autoclip.txt"
    writeCutResults([], str(out))
    assert out.exists() and out.read_text() == ""


def testCreatesMissingOutputDirectory(tmp_path):
    out = tmp_path / "nested" / "dir" / "clip-Autoclip.txt"
    writeCutResults([2.0], str(out))
    assert out.read_text() == "2.0\n"


def testAdobeModeKeepsLegacyCopy(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ADOBE", True)
    monkeypatch.setattr(cs, "WHEREAMIRUNFROM", str(tmp_path))
    out = tmp_path / "renders" / "clip-Autoclip.txt"
    writeCutResults([4.5], str(out))
    legacy = os.path.join(str(tmp_path), "autoclipresults.txt")
    assert out.read_text() == "4.5\n"
    assert open(legacy).read() == "4.5\n"


def testNonAdobeWritesNoLegacyCopy(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ADOBE", False)
    monkeypatch.setattr(cs, "WHEREAMIRUNFROM", str(tmp_path))
    writeCutResults([4.5], str(tmp_path / "clip-Autoclip.txt"))
    assert not os.path.exists(os.path.join(str(tmp_path), "autoclipresults.txt"))
