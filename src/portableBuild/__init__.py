"""The portable build: a bundled interpreter plus the sources, run by build.py.

Standard library only, and nothing here imports the rest of ``src``.
"""

from .builder import buildPortable, outputDirFor
from .settings import BuildSettings, defaultSettings

__all__ = ["BuildSettings", "buildPortable", "defaultSettings", "outputDirFor"]
