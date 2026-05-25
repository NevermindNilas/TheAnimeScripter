"""Tests for src/utils/dependencyHandler.py — profile/requirements resolution.

Pure mapping logic that decides which extra-requirements file an install pulls
based on OS + CUDA support. A wrong mapping here silently installs the wrong
runtime stack, so the table is pinned exactly.
"""

import pytest

from src.utils.dependencyHandler import (
    getDependencyProfile,
    getRequirementsFileForProfile,
    _versionSatisfiesRequirement,
    DEPENDENCY_PROFILE_REQUIREMENTS,
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
        ("Darwin", True, "macos-mps"),   # macOS ignores cuda flag entirely
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
    assert getRequirementsFileForProfile("  WINDOWS-CUDA  ") == "extra-requirements-windows.txt"


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
