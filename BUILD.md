# Build Instructions

## Prerequisites

- Python 3.14 installed and accessible from PATH
- Windows 11, Linux, or Apple Silicon macOS 15 or newer (PyTorch 2.15's MPS backend needs it)

The build does not need FFmpeg on the build machine, and deliberately does not bundle it: FFmpeg is GPL, so shipping it inside a release asset would make TAS a redistributor. TAS itself decodes and encodes through the `nelux` wheel, which carries its own FFmpeg libraries; only YouTube URL inputs need the FFmpeg executables (for yt-dlp), which TAS obtains on first use — Windows and Linux download them, macOS installs them via Homebrew.

## Build Process

### Quick Start
Run this from the repository root to build the portable bundle for the platform you are on:
```sh
python build.py
```

The build uses [requirements.txt](requirements.txt) plus the runtime-specific
requirements files as the source of truth for portable dependency installation.
On macOS the build produces an Apple Silicon (arm64) portable Python bundle. The macOS runtime dependencies are not installed at build time: the bundle is pre-seeded with the `macos-mps` profile, and TAS installs them on first run on the user's machine from
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

Linux and macOS portable builds are started through the bundled launcher script instead:

```sh
./run.sh -h
```

Linux x86_64 releases ship as `TAS-<version>-Linux-x86_64.tar.gz`, which holds the bundle's contents with no top-level folder and no dependency profile:

```sh
mkdir TAS && tar -xzf TAS-<version>-Linux-x86_64.tar.gz -C TAS && cd TAS
./run.sh --download_requirements linux-cuda   # or linux-lite, linux-rocm
```

For end users on Windows, the preferred install flow is the portable bootstrap script hosted on the project site:

```powershell
iwr -useb https://tas.nevermindnilas.dev/install.ps1 | iex
```

It installs the portable bundle into the directory where the script was invoked, creates `TheAnimeScripter.cmd` and `tas.cmd`, and can optionally add that directory to the user PATH with `-AddToPath`.

That form prompts for PATH registration during installation.

## Releases

A release is three archives (`TAS-<version>-Windows.zip`, `-macOS-arm64.zip`, `-Linux-x86_64.tar.gz`) hosted in two places:

- **Update host**, `https://updates.nevermindnilas.dev/engine/stable/`. This is where TAS-Standalone and the Adobe panel download the engine. `index.json` lists every retained release with each file's size and SHA-256. Old releases are pruned automatically; a release stays as long as a current frontend build pins it.
- **GitHub release** `v<version>`, which also carries the Standalone and Adobe builds pinned to that engine version.

The `Build-All-Platforms` workflow builds the archives, creates the GitHub release as a draft, and publishes the archives to the update host once all three platforms have built. Each frontend pins one exact engine version, so every engine release is followed by a release of both frontends pinned to it. The `Sync frontends to release` workflow (every 30 minutes, or run by hand) copies those frontend builds from the update host onto the draft. Publish the draft only after both frontends are attached: Standalone installs older than the update host read their update feed from the release GitHub marks as Latest.

A published version cannot be replaced on the update host. If an archive has to change, release a new version. If a job of the release run fails, use "Re-run failed jobs"; "Re-run all jobs" is refused once the version is on the update host, because a rebuild would not be byte-identical.

The sync job also updates releases that are already published: a frontend-only release replaces that frontend's files on the release of the engine it pins. It fails, and keeps failing every 30 minutes, while the Latest release has no Standalone update feed. On GitHub, `prune-releases.yml` runs when a release is published and removes the `.zip`, `.7z` and `.tar.gz` assets of all but the three newest published releases. That includes the frontends' zips on those older releases, not only the engine archives. The update host keeps every engine version a current frontend build pins.

Two things to keep an eye on:

- GitHub switches scheduled workflows off after 60 days without repository activity. If that happens, frontend-only releases stop reaching GitHub until `Sync frontends to release` is enabled again.
- Rolling a frontend back on the update host to a build that pins an older engine leaves the bad build on the newer GitHub release. The sync job fails while that release is Latest. Mark the older release as Latest (`gh release edit v<older> --latest`) or publish a fixed frontend build; do not delete the frontend files from the Latest release.

## Troubleshooting

- Ensure Python 3.14 is properly installed and added to your system PATH
- On Apple Silicon macOS, YouTube URL inputs need `ffmpeg` and `ffprobe` on PATH (TAS runs `brew install ffmpeg` if Homebrew is present)
- Verify you have sufficient disk space for the build process
