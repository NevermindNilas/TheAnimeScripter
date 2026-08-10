"""Pin every depth backend to the shared WriteBuffer.

Three of them used to encode with `cv2.VideoWriter(fourcc "mp4v")`, which is
not wired to anything the user typed: `--encode_method`, `--custom_encoder`
and `--bit_depth` were all discarded and every run wrote MPEG-4 Part 2. That
covered 25 of the 69 `--depth_method` choices, the whole `*_v3` family
included, and nothing in the log said so.

The check is AST-level on purpose: constructing these classes needs weights, a
GPU and, for the MPS pair, Apple hardware no CI runner has.
"""

import ast
from pathlib import Path

BACKENDS = Path(__file__).resolve().parent.parent / "src" / "depth" / "backends"


def _sources():
    return sorted(p for p in BACKENDS.glob("*.py") if not p.name.startswith("_"))


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _calleeName(call):
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def testNoDepthBackendEncodesWithCv2():
    offenders = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _calls(tree):
            if _calleeName(call) == "VideoWriter":
                offenders.append(f"{path.name}:{call.lineno}")
    assert not offenders, (
        "depth backends must encode through WriteBuffer so --encode_method, "
        f"--custom_encoder and --bit_depth are honored; found: {offenders}"
    )


def testEveryDepthBackendClassBuildsAWriteBuffer():
    """A backend that takes `encode_method` must actually hand it to a writer."""
    missing = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            init = next(
                (
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                ),
                None,
            )
            if init is None:
                # A subclass that inherits __init__ inherits its writer too.
                continue
            takesEncodeMethod = any(
                arg.arg == "encode_method" for arg in init.args.args
            )
            if not takesEncodeMethod:
                continue
            buildsWriter = any(
                _calleeName(call) in ("WriteBuffer", "createWriteBuffer")
                for call in _calls(init)
            )
            if not buildsWriter:
                missing.append(f"{path.name}:{node.name}")
    assert not missing, (
        f"depth backends accepting encode_method but never building a writer "
        f"with it: {missing}"
    )
