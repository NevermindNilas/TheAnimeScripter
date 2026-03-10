# Build Instructions

## Prerequisites

- Python 3.13 installed and accessible from PATH
- uv installed and accessible from PATH
- Windows 11

## Build Process

### Quick Start
To build the project on Windows, run:
```sh
python build.py
```

The build uses [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock) as the
canonical source of truth for core and runtime dependencies.

The portable bundle now ships with `uv.exe` alongside `python.exe` so runtime
dependency syncs can be reproduced from the locked metadata without bootstrapping pip.

### Output Location
After successful compilation, the executable files will be located in:
```
\dist-portable\main\
```

## Runtime Dependencies

TAS automatically downloads and installs additional dependencies as needed:
- Core dependencies are locked through `pyproject.toml` + `uv.lock`
- Runtime profiles are exported from the lockfile and synced with `uv pip sync`

All downloaded dependencies are placed in the same directory as the executable.

## Usage

For detailed usage instructions, refer to:
- [PARAMETERS.MD](PARAMETERS.MD) for comprehensive parameter documentation
- Or run the help command:
```sh
.\python.exe .\main.py -h
```

For end users on Windows, the preferred install flow is the portable bootstrap script in the repository root:

```powershell
iwr -useb https://raw.githubusercontent.com/NevermindNilas/TheAnimeScripter/main/install.ps1 | iex
```

It installs the portable bundle into the directory where the script was invoked, creates `TheAnimeScripter.cmd` and `tas.cmd`, and can optionally add that directory to the user PATH with `-AddToPath`.

That form prompts for PATH registration during installation.

## Troubleshooting

- Ensure Python 3.13 is properly installed and added to your system PATH
- Verify you have sufficient disk space for the build process