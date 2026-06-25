def providedCliOptions(argv):
    """Return normalized long-option names provided on the command line."""
    provided = set()
    for arg in argv:
        if arg.startswith("--"):
            provided.add(arg[2:].split("=", 1)[0].replace("-", "_"))
    return provided


def wasProvided(args, optionName, cliOptions):
    if optionName in cliOptions:
        return True
    return hasattr(args, "_json_keys") and optionName in args._json_keys
