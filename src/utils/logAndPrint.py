"""
Logging and Print Utilities

Provides unified functions for both console output and file logging.
Ensures consistent message formatting and logging throughout the application.
"""

import logging
import sys
import os

# ANSI escape codes - works on modern terminals (Windows 10+, Linux, macOS)
_RESET = "\033[0m"
_BOLD = "\033[1m"

_COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "light_blue": "\033[94m",
}

# Detect whether the output supports ANSI colors
def _supportsColor():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        # Windows 10 build 14393+ supports ANSI natively
        try:
            ver = sys.getwindowsversion()
            return ver.major >= 10 and ver.build >= 14393
        except Exception:
            return False
    return True


_COLOR_ENABLED = _supportsColor()


def _ansi(text, color_code):
    if not _COLOR_ENABLED:
        return str(text)
    return f"{color_code}{text}{_RESET}"


_VERBOSE = False
_QUIET = False


def setVerbose(verbose: bool) -> None:
    """
    Set global verbose mode for detailed logging.

    Args:
        verbose (bool): Enable verbose mode
    """
    global _VERBOSE
    _VERBOSE = verbose


def setQuiet(quiet: bool) -> None:
    """
    Set global quiet mode for minimal console output.

    Args:
        quiet (bool): Enable quiet mode
    """
    global _QUIET
    _QUIET = quiet


def logAndPrint(message: str, colorFunc: str = "cyan", level: str = "INFO") -> None:
    """
    Print a colored message to console and log it to file.

    Args:
        message (str): Message to display and log
        colorFunc (str): Color function name ('cyan', 'red', 'yellow', 'green')
        level (str): Log level ('INFO', 'WARNING', 'ERROR', 'SUCCESS', 'DEBUG')
    """
    colorFunctions = {"cyan": cyan, "red": red, "yellow": yellow, "green": green}

    icons = {"INFO": "i", "WARNING": "!", "ERROR": "x", "SUCCESS": "+", "DEBUG": "?"}

    icon = icons.get(level.upper(), "i")
    if not _QUIET:
        if colorFunc in colorFunctions:
            formatted_message = f"{icon} {message}"
            print(colorFunctions[colorFunc](formatted_message))
        else:
            print(f"{icon} {message}")

    if level.upper() == "ERROR":
        logging.error(message)
    elif level.upper() == "WARNING":
        logging.warning(message)
    elif level.upper() == "DEBUG" and _VERBOSE:
        logging.debug(message)
    else:
        logging.info(message)


def logInfo(message: str) -> None:
    """Log informational message."""
    logAndPrint(message, "cyan", "INFO")


def logSuccess(message: str) -> None:
    """Log success message."""
    logAndPrint(message, "green", "SUCCESS")


def logWarning(message: str) -> None:
    """Log warning message."""
    logAndPrint(message, "yellow", "WARNING")


def logError(message: str) -> None:
    """Log error message."""
    logAndPrint(message, "red", "ERROR")


def logDebug(message: str) -> None:
    """Log debug message (only shown in verbose mode)."""
    if _VERBOSE:
        logAndPrint(message, "cyan", "DEBUG")


def printSectionHeader(title: str, char: str = "=") -> None:
    """
    Print a formatted section header for better visual organization.

    Args:
        title (str): Section title
        char (str): Character to use for the border (default: '=')
    """
    width = 80
    if not _QUIET:
        border = char * width
        print(cyan(border))
        print(cyan(f"{title.center(width)}"))
        print(cyan(border))

    logging.info(f"\n{border}")
    logging.info(f"{title.center(width)}")
    logging.info(f"{border}\n")


def printSubsectionHeader(title: str) -> None:
    """
    Print a formatted subsection header.

    Args:
        title (str): Subsection title
    """
    if not _QUIET:
        print(cyan(f"\n{'─' * 80}"))
        print(cyan(f"  {title}"))
        print(cyan(f"{'─' * 80}"))

    logging.info(f"\n{'─' * 80}")
    logging.info(f"  {title}")
    logging.info(f"{'─' * 80}")


def coloredPrint(message: str, colorFunc: str = "cyan") -> None:
    """
    Print a colored message to console only (no logging).

    Args:
        message (str): Message to display
        colorFunc (str): Color function name ('cyan', 'red', 'yellow', 'green')
    """
    if _QUIET:
        return

    colorFunctions = {"cyan": cyan, "red": red, "yellow": yellow, "green": green}

    if colorFunc in colorFunctions:
        print(colorFunctions[colorFunc](message))
    else:
        print(message)


def green(text):
    """Format text in green color."""
    return _ansi(text, _COLORS["green"])


def red(text):
    """Format text in red color."""
    return _ansi(text, _COLORS["red"])


def yellow(text):
    """Format text in yellow color."""
    return _ansi(text, _COLORS["yellow"])


def blue(text):
    """Format text in blue color."""
    return _ansi(text, _COLORS["blue"])


def magenta(text):
    """Format text in magenta color."""
    return _ansi(text, _COLORS["magenta"])


def cyan(text):
    """Format text in cyan color."""
    return _ansi(text, _COLORS["cyan"])


def lightBlue(text):
    """Format text in light blue color."""
    return _ansi(text, _COLORS["light_blue"])


def rainbow(text):
    """Format text with rainbow colors, cycling through different colors per character."""
    if not _COLOR_ENABLED:
        return str(text)
    colors = ["red", "yellow", "green", "blue", "magenta", "cyan"]
    result = ""
    for i, char in enumerate(text):
        result += f"{_COLORS[colors[i % len(colors)]]}{char}"
    return f"{result}{_RESET}"


def bold(text):
    """Format text in bold style."""
    return _ansi(text, _BOLD)
