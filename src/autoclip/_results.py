"""Shared cut-list writer for the autoclip drivers."""

import logging
import os
import shutil

import src.constants as cs
from src.infra.logAndPrint import logAndPrint


def writeCutResults(cuts, outputPath):
    """Write cut timestamps (seconds, one per line) to ``outputPath``.

    The path is the per-video output manufactured by inputOutputHandler, so a
    batch writes one results file per input instead of clobbering a single
    fixed ``autoclipresults.txt`` in the install directory (which is read-only
    for a Program Files install and ignored ``--output`` entirely).

    The After Effects panel historically read the fixed install-dir path, so
    in ADOBE mode a legacy copy is kept there as well.
    """
    outDir = os.path.dirname(outputPath)
    if outDir:
        os.makedirs(outDir, exist_ok=True)

    with open(outputPath, "w") as f:
        for i, t in enumerate(cuts):
            logging.info(f"Scene {i + 1}: cut at {t:.3f}s")
            f.write(f"{t}\n")

    if cs.ADOBE:
        legacyPath = os.path.join(cs.WHEREAMIRUNFROM, "autoclipresults.txt")
        if os.path.abspath(legacyPath) != os.path.abspath(outputPath):
            try:
                shutil.copyfile(outputPath, legacyPath)
            except OSError as e:
                logging.warning(f"Could not update legacy autoclipresults.txt: {e}")

    logAndPrint(f"AutoClip wrote {len(cuts)} cuts to {outputPath}", "green")
