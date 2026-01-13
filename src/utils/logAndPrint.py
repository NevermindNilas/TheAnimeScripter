"""
Logging and Print Utilities

Provides unified functions for both console output and file logging.
Ensures consistent message formatting and logging throughout the application.
"""

import logging
from colored import fg, attr


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

    icons = {"INFO": "ℹ", "WARNING": "⚠", "ERROR": "✗", "SUCCESS": "✓", "DEBUG": "🔍"}

    icon = icons.get(level.upper(), "ℹ")
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


"""
Colored Terminal Output Utilities

Provides functions for colored terminal text output using the colored library.
Used throughout the application for user-friendly console feedback.
"""


def green(text):
    """
    Format text in green color for success messages.

    Args:
        text (str): Text to colorize

    Returns:
        str: Green-colored text with reset attributes
    """
    return "%s%s%s" % (fg("green"), text, attr("reset"))


def red(text):
    """
    Format text in red color for error messages.

    Args:
        text (str): Text to colorize

    Returns:
        str: Red-colored text with reset attributes
    """
    return "%s%s%s" % (fg("red"), text, attr("reset"))


def yellow(text):
    """
    Format text in yellow color for warning messages.

    Args:
        text (str): Text to colorize

    Returns:
        str: Yellow-colored text with reset attributes
    """
    return "%s%s%s" % (fg("yellow"), text, attr("reset"))


def blue(text):
    """
    Format text in blue color for informational messages.

    Args:
        text (str): Text to colorize

    Returns:
        str: Blue-colored text with reset attributes
    """
    return "%s%s%s" % (fg("blue"), text, attr("reset"))


def magenta(text):
    """
    Format text in magenta color.

    Args:
        text (str): Text to colorize

    Returns:
        str: Magenta-colored text with reset attributes
    """
    return "%s%s%s" % (fg("magenta"), text, attr("reset"))


def cyan(text):
    """
    Format text in cyan color for general information.

    Args:
        text (str): Text to colorize

    Returns:
        str: Cyan-colored text with reset attributes
    """
    return "%s%s%s" % (fg("cyan"), text, attr("reset"))


def lightBlue(text):
    """
    Format text in light blue color.

    Args:
        text (str): Text to colorize

    Returns:
        str: Light blue-colored text with reset attributes
    """
    return "%s%s%s" % (fg("light_blue"), text, attr("reset"))


def rainbow(text):
    """
    Format text with rainbow colors, cycling through different colors per character.

    Args:
        text (str): Text to colorize

    Returns:
        str: Rainbow-colored text with reset attributes
    """
    colors = ["red", "yellow", "green", "blue", "magenta", "cyan"]
    coloredText = ""
    for i, char in enumerate(text):
        color = colors[i % len(colors)]
        coloredText += "%s%s%s" % (fg(color), char, attr("reset"))
    return coloredText


def bold(text):
    """
    Format text in bold style.

    Args:
        text (str): Text to make bold

    Returns:
        str: Bold text with reset attributes
    """
    return "%s%s%s" % (attr("bold"), text, attr("reset"))
