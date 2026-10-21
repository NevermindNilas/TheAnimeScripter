"""
Global Constants Configuration

These constants should not change their values once runtime arguments are prepared.
They're defined here to avoid populating the code with excessive arguments
and to improve code readability.
"""

# Core system paths
WHEREAMIRUNFROM: str = ""  # Path to the main script directory
SYSTEM: str = ""  # Operating system identifier (Windows/Linux/macOS)
LOG_PATH: str = ""  # Backend log file path initialized by main()

# FFmpeg executable paths
FFMPEGPATH: str = ""  # Path to FFmpeg executable
FFPROBEPATH: str = ""  # Path to FFprobe executable
METADATAPATH: str = ""  # Path to metadata configuration file

# Feature flags
ADOBE: bool = False  # Enable Adobe After Effects compatibility mode
AUDIO: bool = True  # Audio passthrough for the video currently being processed
# The run-wide audio intent, latched once CLI validation has applied every
# "this mode disables audio" rule. getVideoMetadata ANDs the per-video probe
# into cs.AUDIO for each video, so without a separate record of the intent the
# first silent video in a batch turned audio off for every video after it.
AUDIO_REQUESTED: bool = True

# The current input's probed metadata, set by getVideoMetadata.saveMetadata once
# per video. The writers' colour decision used to round-trip through
# metadata.json at a fixed install-dir path -- identical for every TAS process
# on the machine -- so a second run (routine in the Adobe Edition, which
# relaunches TAS while rendering) could overwrite it between one run's write and
# that run's read, and a BT.2020 source got encoded and tagged bt709 or vice
# versa, silently. Reading in-process keeps the decision inside the run that
# probed it. ``None`` means "not probed yet" and falls back to METADATAPATH;
# it must not be ``{}``, which would suppress the file read for every caller.
PROBED_METADATA: dict | None = None
