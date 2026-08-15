"""Shared pytest fixtures.

``src/constants.py`` holds run-wide state that CLI validation latches once and
the rest of the process reads. Inside a pytest session there is no "once": a
test that sets one of those globals leaves it set for every test after it, and
the resulting failure surfaces somewhere unrelated. Reset the mutable ones
between tests so ordering cannot decide the outcome.
"""

import pytest

import src.constants as cs

# Globals a test may legitimately set, paired with the value that means "unset".
# METADATAPATH is deliberately absent: many tests point it at a tmp file and
# already restore it through monkeypatch.
_RESETTABLE = {
    "PROBED_METADATA": None,
    "OUTPUT_SCALE_WIDTH": None,
    "OUTPUT_SCALE_HEIGHT": None,
}


@pytest.fixture(autouse=True)
def resetRuntimeConstants():
    saved = {name: getattr(cs, name) for name in _RESETTABLE}
    for name, unset in _RESETTABLE.items():
        setattr(cs, name, unset)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(cs, name, value)
