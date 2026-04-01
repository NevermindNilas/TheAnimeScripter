# Build Instructions

## Prerequisites

- Python 3.13 installed and accessible from PATH
- Windows 11

## Build Process

### Quick Start
To build the project on Windows, run:
```sh
python build.py
```

The build uses [requirements.txt](requirements.txt) plus the runtime-specific
requirements files as the source of truth for portable dependency installation.

[pyproject.toml](pyproject.toml) remains in the repository as project metadata,
but the portable build and runtime dependency handling are driven by pip.

### Output Location
After successful compilation, the executable files will be located in:
```
\dist-portable\main\
```

## Runtime Dependencies

TAS automatically downloads and installs additional dependencies as needed:
- Core dependencies are installed from `requirements.txt`
- Runtime profiles are installed from the matching `extra-requirements-*.txt` file with pip

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