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
