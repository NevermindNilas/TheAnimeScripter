"""
Startup Performance Profiler

Measures import times and initialization overhead to identify bottlenecks.
"""

import sys
import time
from importlib import import_module


def profile_imports():
    """Profile import times for key modules."""
    modules_to_test = [
        "torch",
        "torchvision",
        "numpy",
        "onnxruntime",
        "timm",
        "einops",
        "rich",
        "argparse",
    ]

    results = []

    for module_name in modules_to_test:
        if module_name in sys.modules:
            print(f"⚠ {module_name} already imported")
            continue

        start = time.perf_counter()
        try:
            import_module(module_name)
            elapsed = (time.perf_counter() - start) * 1000
            results.append((module_name, elapsed))
            print(f"✓ {module_name}: {elapsed:.2f}ms")
        except ImportError as e:
            print(f"✗ {module_name}: Import failed - {e}")

    return results


def measure_startup_phases():
    """Measure different phases of startup."""
    phases = {}

    # Phase 1: Argument parsing
    start = time.perf_counter()
    from src.utils.argumentsChecker import createParser

    elapsed = (time.perf_counter() - start) * 1000
    phases["argument_parser_import"] = elapsed
    print(f"\nArgument parser import: {elapsed:.2f}ms")

    # Phase 2: Constants/config
    start = time.perf_counter()
    import src.constants as cs

    elapsed = (time.perf_counter() - start) * 1000
    phases["constants_import"] = elapsed
    print(f"Constants import: {elapsed:.2f}ms")

    return phases


if __name__ == "__main__":
    print("=" * 60)
    print("TAS Startup Performance Profiler")
    print("=" * 60)

    # Overall timing
    overall_start = time.perf_counter()

    print("\n[Phase 1] Import Profiling")
    print("-" * 60)
    import_results = profile_imports()

    print("\n[Phase 2] Startup Phases")
    print("-" * 60)
    phase_results = measure_startup_phases()

    overall_elapsed = (time.perf_counter() - overall_start) * 1000

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if import_results:
        print("\nTop 5 Slowest Imports:")
        sorted_imports = sorted(import_results, key=lambda x: x[1], reverse=True)[:5]
        for module, ms in sorted_imports:
            print(f"  {module}: {ms:.2f}ms")

    print(f"\nTotal profiling time: {overall_elapsed:.2f}ms")
