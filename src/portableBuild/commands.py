"""The one place the build starts another program."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


def runCommand(
    command: Sequence[str],
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run ``command`` without a shell, output inherited; a non-zero exit raises."""
    try:
        subprocess.run(command, check=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as error:
        print(f"Error while running command {command}: {error}")
        raise
