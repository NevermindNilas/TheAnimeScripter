"""
Colored Terminal Output Utilities

Provides functions for colored terminal text output using the colored library.
Used throughout the application for user-friendly console feedback.
"""

from colored import fg, attr


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
