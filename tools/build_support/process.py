import subprocess
from pathlib import Path


def run_subprocess(
    command,
    shell: bool = False,
    cwd: Path | str | None = None,
    stdout=None,
    env=None,
) -> None:
    try:
        subprocess.run(
            command,
            shell=shell,
            check=True,
            cwd=cwd,
            stdout=stdout,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error while running command {command}: {e}")
        raise


def run_subprocess_result(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)
