"""
Global Constants Configuration

These constants should not change their values once past argumentsChecker.py.
They're defined here to avoid populating the code with excessive arguments
and to improve code readability.
"""

# Core settings
WHEREAMIRUNFROM: str = ""  # The path to the main script (default: 'Wherever TAS is ran from without main.py!')
MAINPATH: str = ""  # The path to logging and debugging (default: 'C:\Users\nilas\AppData\Roaming\TheAnimeScripter')
SYSTEM: str = ""  # The operating system (default: 'Windows')
FFMPEGPATH: str = ""  # The path to the FFmpeg executable (default: 'C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg.exe')
FFPROBEPATH: str = ""  # The path to the FFprobe executable (default: 'C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffprobe.exe')
MPVPATH: str = ""  # The path to the MPV executable (default: 'C:\Users\nilas\AppData\Roaming\TheAnimeScripter\mpv.exe')

# Feature flags
ADOBE: bool = False  # Enables Adobe compatibility (logs progress, frame data to JSON)
AUDIO: bool = (
    True  # Enables audio processing, also acts as a "is there any audio?" flag
)
