"""Pin --stabilize_method's dispatch in the standalone factory.

The dut/classic split lives in one lazy-import branch inside
``standalone.stabilize()``; nothing else exercises it without a GPU and
weights, so this guards the wiring the way test_runOutcome guards the
outcome contract: structurally.
"""

import ast
from pathlib import Path

STANDALONE = (
    Path(__file__).resolve().parent.parent / "src" / "factories" / "standalone.py"
)


def _stabilizeFunction():
    tree = ast.parse(STANDALONE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "stabilize":
            return node
    raise AssertionError("standalone.py no longer defines stabilize()")


def _importedModules(nodes):
    modules = set()
    for node in ast.walk(ast.Module(body=list(nodes), type_ignores=[])):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def testStabilizeDispatchesOnMethod():
    fn = _stabilizeFunction()
    branches = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and "stabilizeMethod" in ast.dump(node.test)
        and "'dut'" in ast.dump(node.test)
    ]
    assert branches, (
        "stabilize() no longer branches on self.stabilizeMethod == 'dut'; "
        "--stabilize_method dut would silently run the classic path."
    )
    branch = branches[0]
    assert "src.stabilize.dutStabilizer" in _importedModules(branch.body), (
        "the dut branch must import the DUT driver"
    )
    assert "src.stabilize.stabilize" in _importedModules(branch.orelse), (
        "the non-dut branch must keep importing the classic driver"
    )


def testStabilizeStillReadsProcessingErrorBack():
    # Whichever driver ran, the factory must hand its processingError to
    # main.py — same contract test_runOutcome pins for all video capabilities.
    fn = _stabilizeFunction()
    assigns = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "processingError"
            for t in node.targets
        )
    ]
    assert assigns, "stabilize() no longer copies processingError back"
