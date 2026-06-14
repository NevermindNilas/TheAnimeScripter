"""Tests for src/utils/dependencyHandler.py — profile/requirements resolution.

Pure mapping logic that decides which extra-requirements file an install pulls
based on OS + CUDA support. A wrong mapping here silently installs the wrong
runtime stack, so the table is pinned exactly.
"""

import pytest

import src.utils.dependencyHandler as dh
from src.utils.dependencyHandler import (
    _MAXINE_UNUSED_LIBS,
    DEPENDENCY_PROFILE_REQUIREMENTS,
    _versionSatisfiesRequirement,
    getDependencyProfile,
    getRequirementsFileForProfile,
    pruneMaxineUnusedLibs,
)

# --------------------------------------------------------------------------- #
# getDependencyProfile
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "system,cuda,expected",
    [
        ("Windows", True, "windows-cuda"),
        ("Windows", False, "windows-lite"),
        ("Linux", True, "linux-cuda"),
        ("Linux", False, "linux-lite"),
        ("Darwin", True, "macos-mps"),  # macOS ignores cuda flag entirely
        ("Darwin", False, "macos-mps"),
    ],
)
def testProfileSelection(system, cuda, expected):
    assert getDependencyProfile(system, cuda) == expected


def testProfileIsCaseInsensitive():
    assert getDependencyProfile("WINDOWS", True) == "windows-cuda"


# --------------------------------------------------------------------------- #
# getRequirementsFileForProfile
# --------------------------------------------------------------------------- #


def testEveryProfileMapsToARequirementsFile():
    for profile in DEPENDENCY_PROFILE_REQUIREMENTS:
        assert getRequirementsFileForProfile(profile).startswith("extra-requirements")


def testProfileLookupTrimsAndLowercases():
    assert (
        getRequirementsFileForProfile("  WINDOWS-CUDA  ")
        == "extra-requirements-windows.txt"
    )


def testUnknownProfileRaisesWithValidOptions():
    with pytest.raises(ValueError) as exc:
        getRequirementsFileForProfile("bogus")
    # Error message must list the legal profiles to be actionable.
    assert "windows-cuda" in str(exc.value)


# --------------------------------------------------------------------------- #
# _versionSatisfiesRequirement
# --------------------------------------------------------------------------- #


def testExactPinSatisfied():
    assert _versionSatisfiesRequirement("numpy==1.2.3", "1.2.3") is True


def testLowerBoundSatisfied():
    assert _versionSatisfiesRequirement("numpy>=1.0", "2.0") is True


def testLowerBoundViolated():
    assert _versionSatisfiesRequirement("numpy>=2.0", "1.0") is False


def testBarePackageAlwaysSatisfied():
    # No specifier -> any installed version counts.
    assert _versionSatisfiesRequirement("numpy", "9.9") is True


def testUnparseableSpecifierIsLenient():
    # Bad input must not crash the dependency check; it errs on "satisfied".
    assert _versionSatisfiesRequirement("!!!not a requirement!!!", "1.0") is True


# --------------------------------------------------------------------------- #
# pruneMaxineUnusedLibs — trims the TensorRT libs the Maxine VSR path never uses
# --------------------------------------------------------------------------- #


def _fakeLibsDir(tmp_path):
    """Build a stand-in nvvfx/libs with the TRT trio + one keep-file."""
    libs = tmp_path / "nvvfx" / "libs"
    libs.mkdir(parents=True)
    for name in _MAXINE_UNUSED_LIBS:
        (libs / name).write_bytes(b"x" * 1024)
    # A file VSR actually needs must survive the prune.
    (libs / "nvngx_vsr.dll").write_bytes(b"keepme")
    return libs


def testPruneRemovesTrtTrioAndReportsBytes(tmp_path, monkeypatch):
    libs = _fakeLibsDir(tmp_path)
    monkeypatch.setattr(dh, "_locateNvvfxLibsDir", lambda: libs)

    removed, freed = pruneMaxineUnusedLibs()

    # Only the platform-matching subset exists as real names, but the fake dir
    # wrote every entry in the table, so all are removed.
    assert removed == len(_MAXINE_UNUSED_LIBS)
    assert freed == removed * 1024
    for name in _MAXINE_UNUSED_LIBS:
        assert not (libs / name).exists()


def testPruneKeepsNonTargetLibs(tmp_path, monkeypatch):
    libs = _fakeLibsDir(tmp_path)
    monkeypatch.setattr(dh, "_locateNvvfxLibsDir", lambda: libs)

    pruneMaxineUnusedLibs()

    # The VSR model DLL must never be touched.
    assert (libs / "nvngx_vsr.dll").read_bytes() == b"keepme"


def testPruneIsIdempotent(tmp_path, monkeypatch):
    libs = _fakeLibsDir(tmp_path)
    monkeypatch.setattr(dh, "_locateNvvfxLibsDir", lambda: libs)

    pruneMaxineUnusedLibs()
    removed, freed = pruneMaxineUnusedLibs()

    assert removed == 0
    assert freed == 0


def testPruneNoopWhenPackageAbsent(monkeypatch):
    # No nvidia-vfx installed -> locate returns None -> clean no-op, no raise.
    monkeypatch.setattr(dh, "_locateNvvfxLibsDir", lambda: None)
    assert pruneMaxineUnusedLibs() == (0, 0)
