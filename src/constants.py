"""
Global Constants Configuration

These constants should not change their values once past argumentsChecker.py.
They're defined here to avoid populating the code with excessive arguments
and to improve code readability.
"""

# Core system paths
WHEREAMIRUNFROM: str = ""  # Path to the main script directory
MAINPATH: str = ""  # Path for logging and application data storage
SYSTEM: str = ""  # Operating system identifier (Windows/Linux/macOS)

# FFmpeg executable paths
FFMPEGPATH: str = ""  # Path to FFmpeg executable
FFPROBEPATH: str = ""  # Path to FFprobe executable
METADATAPATH: str = ""  # Path to metadata configuration file

# Feature flags
ADOBE: bool = False  # Enable Adobe After Effects compatibility mode
AUDIO: bool = True  # Enable audio processing and detection
