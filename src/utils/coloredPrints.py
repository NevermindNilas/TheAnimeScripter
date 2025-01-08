from colored import fg, attr


def green(text):
    return "%s%s%s" % (fg("green"), text, attr("reset"))


def red(text):
    return "%s%s%s" % (fg("red"), text, attr("reset"))


def yellow(text):
    return "%s%s%s" % (fg("yellow"), text, attr("reset"))


def blue(text):
    return "%s%s%s" % (fg("blue"), text, attr("reset"))


def magenta(text):
    return "%s%s%s" % (fg("magenta"), text, attr("reset"))


def cyan(text):
    return "%s%s%s" % (fg("cyan"), text, attr("reset"))


def lightBlue(text):
    return "%s%s%s" % (fg("light_blue"), text, attr("reset"))


def rainbow(text):
    colors = ["red", "yellow", "green", "blue", "magenta", "cyan"]
    coloredText = ""
    for i, char in enumerate(text):
        color = colors[i % len(colors)]
        coloredText += "%s%s%s" % (fg(color), char, attr("reset"))
    return coloredText


def bold(text):
    return "%s%s%s" % (attr("bold"), text, attr("reset"))
