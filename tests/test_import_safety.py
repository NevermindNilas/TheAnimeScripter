import py_compile
import subprocess
import sys
from pathlib import Path


def testCliAndInfraEntryModulesCompile():
    root = Path(__file__).resolve().parents[1]
    modules = [
        root / "main.py",
        root / "src" / "cli" / "config.py",
        root / "src" / "cli" / "parser.py",
        root / "src" / "cli" / "validation.py",
        root / "src" / "cli" / "validator.py",
        root / "src" / "infra" / "backendFallback.py",
        root / "src" / "infra" / "dependencyHandler.py",
        root / "src" / "infra" / "isCudaInit.py",
        root / "src" / "io" / "inputNormalization.py",
    ]
    for module in modules:
        py_compile.compile(str(module), doraise=True)


def testLightCliInfraImportsDoNotLoadRuntimeStacks():
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib
import sys

for module in (
    "src.cli.parser",
    "src.cli.config",
    "src.cli.validator",
    "src.cli.validation",
    "src.infra.backendFallback",
    "src.infra.dependencyHandler",
    "src.infra.isCudaInit",
    "src.io.inputNormalization",
):
    importlib.import_module(module)

loaded = sorted({"torch", "cv2", "nelux"} & set(sys.modules))
if loaded:
    raise SystemExit(f"heavy runtime modules imported at startup: {loaded}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
