# Build Instructions

## Prerequisites

- Python 3.14 installed and accessible from PATH
- Windows 11, Linux, or Apple Silicon macOS

The build does not need FFmpeg on the build machine, and deliberately does not bundle it: FFmpeg is GPL, so shipping it inside a release asset would make TAS a redistributor. TAS itself decodes and encodes through the `nelux` wheel, which carries its own FFmpeg libraries; only YouTube URL inputs need the FFmpeg executables (for yt-dlp), which TAS obtains on first use — Windows and Linux download them, macOS installs them via Homebrew.

## Build Process

### Quick Start
To build the project for the current platform, run:
```sh
python build.py
```

The build uses [requirements.txt](requirements.txt) plus the runtime-specific
requirements files as the source of truth for portable dependency installation.
On macOS, `build.py` creates an Apple Silicon portable Python bundle and installs
`extra-requirements-macos.txt`.

### Output Location
After successful compilation, the executable files will be located in:
```
dist-portable/main/
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

On Linux/macOS portable builds, use:

```sh
./run.sh -h
```

For end users on Windows, the preferred install flow is the portable bootstrap script hosted on the project site:

```powershell
iwr -useb https://tas.nevermindnilas.dev/install.ps1 | iex
```

It installs the portable bundle into the directory where the script was invoked, creates `TheAnimeScripter.cmd` and `tas.cmd`, and can optionally add that directory to the user PATH with `-AddToPath`.

That form prompts for PATH registration during installation.

## Troubleshooting

- Ensure Python 3.14 is properly installed and added to your system PATH
- On Apple Silicon macOS, YouTube URL inputs need `ffmpeg` and `ffprobe` on PATH (TAS runs `brew install ffmpeg` if Homebrew is present)
- Verify you have sufficient disk space for the build process
