"""Guards against a half-applied bundled-Python bump.

`tools/build_support/context.py` is the single source of truth for the bundled
interpreter, but the version is repeated in three build workflows and in
`CLAUDE.md`, and the download URLs are assembled from string literals. The
fixtures in `test_build.py`/`test_build_refactor.py` only feed the constructor,
so they pass no matter what those copies say -- mutating them to a bogus version
leaves the suite green. These tests fail instead.
"""

import re
from pathlib import Path

from tools.build_support.context import create_build_context

ROOT = Path(__file__).resolve().parents[1]
BUILD_WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("Build-*.yaml"))
SUPPORT_MODULES = sorted((ROOT / "tools" / "build_support").glob("*.py"))

# "3.14.6" and friends, but not a bare "3.14" or a version-looking pin such as
# the uv "0.11.28" that lives beside it.
PYTHON_VERSION = re.compile(r"\b3\.\d+\.\d+\b")
STANDALONE_TAG = re.compile(r"\b20\d{6}\b")


def _context():
    return create_build_context(ROOT)


def test_build_workflows_pin_the_bundled_python_version():
    """Every `python-version:` in a build workflow must match context.py."""
    expected = _context().python_version
    assert BUILD_WORKFLOWS, "no Build-*.yaml workflows found"

    seen = 0
    for workflow in BUILD_WORKFLOWS:
        for line, pinned in enumerate(
            re.findall(r'python-version:\s*"([^"]+)"', workflow.read_text()), start=1
        ):
            # The commented-out lite jobs still pin an old 3.13; only live
            # 3.14-series pins are part of this contract.
            if not pinned.startswith("3.14"):
                continue
            seen += 1
            assert pinned == expected, (
                f"{workflow.name} pins python-version {pinned!r}, but "
                f"tools/build_support/context.py says {expected!r} (pin #{line})"
            )

    assert seen, "no 3.14 python-version pins found in the build workflows"


def test_context_is_the_only_place_holding_the_version():
    """No build_support module may hardcode a version or standalone tag."""
    context = _context()

    for module in SUPPORT_MODULES:
        if module.name == "context.py":
            continue
        text = module.read_text()
        assert not PYTHON_VERSION.search(text), (
            f"{module.name} hardcodes a Python version; it must read "
            f"context.python_version so a bump stays in one place"
        )
        assert not STANDALONE_TAG.search(text), (
            f"{module.name} hardcodes a python-build-standalone tag; it must "
            f"read context.standalone_release"
        )

    source = (ROOT / "tools" / "build_support" / "context.py").read_text()
    assert context.python_version in source
    assert context.standalone_release in source


def test_claude_md_documents_the_bundled_python_version():
    """CLAUDE.md states the bundled version; keep it from going stale."""
    expected = _context().python_version
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    stated = PYTHON_VERSION.findall(text)
    assert stated, "CLAUDE.md no longer states a bundled Python patch version"
    assert all(version == expected for version in stated), (
        f"CLAUDE.md mentions {sorted(set(stated))}, expected {expected!r}"
    )
