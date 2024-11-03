from alive_progress import alive_bar


def progressBarLogic(totalFrames: int) -> alive_bar:
    """
    This function creates a progress bar for the given range of frames.
    The scope of this is to be less repetitive and more modular.

    Args:
        totalFrames (int): The total number of frames to process

    Returns:
        alive_bar: The progress bar

    """
    return alive_bar(
        total=totalFrames,
        title="Processing Frame: ",
        length=30,
        stats="| {rate} | ETA: {eta}",
        elapsed="Elapsed Time: {elapsed}",
        monitor=" {count}/{total} | [{percent:.0%}] | ",
        unit="frames",
        spinner=None,
    )


def progressBarDownloadLogic(totalData: int, title: str) -> alive_bar:
    """
    This function creates a progress bar for the given range of frames.
    The scope of this is to be less repetitive and more modular.

    Args:
        totalData (int): The total number of frames to process
        title (str): The title of the progress bar

    Returns:
        alive_bar: The progress bar

    """
    return alive_bar(
        total=totalData,
        title=title,
        length=30,
        stats="| {rate} | ETA: {eta}",
        elapsed="Elapsed Time: {elapsed}",
        monitor=" {count}/{total} | [{percent:.0%}] | ",
        unit="MB",
        spinner=None,
    )
