# Build Instructions

## Prerequisites

- Python 3.13 installed and accessible from PATH
- Windows 10 or Windows 11

## Build Process

### Quick Start
To build the project on Windows, run:
```sh
python build.py
```

### Output Location
After successful compilation, the executable files will be located in:
```
\dist-portable\main\
```

## Runtime Dependencies

TAS automatically downloads and installs additional dependencies as needed:
- See `extra-requirements-windows.txt` for full version dependencies
- See `extra-requirements-windows-lite.txt` for minimal version dependencies

All downloaded dependencies are placed in the same directory as the executable.

## Usage

For detailed usage instructions, refer to:
- [PARAMETERS.MD](PARAMETERS.MD) for comprehensive parameter documentation
- Or run the help command:
```sh
.\python.exe .\main.py -h
```

## Troubleshooting

- Ensure Python 3.13 is properly installed and added to your system PATH
- Verify you have sufficient disk space for the build process