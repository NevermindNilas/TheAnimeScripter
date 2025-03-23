"""
Global Constants Configuration

These constants should not change their values once past argumentsChecker.py.
They're defined here to avoid populating the code with excessive arguments
and to improve code readability.
"""

# Core settings
MAINPATH: str = ""  # The path to the project directory (default: 'C:\Users\nilas\AppData\Roaming\TheAnimeScripter')
SYSTEM: str = ""  # The operating system (default: 'Windows')
FFMPEGPATH: str = ""  # The path to the FFmpeg executable (default: 'C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg.exe')

# Feature flags
ADOBE: bool = False  # Enables Adobe compatibility (logs progress, frame data to JSON)
AUDIO: bool = (
    True  # Enables audio processing, also acts as a "is there any audio?" flag
)
