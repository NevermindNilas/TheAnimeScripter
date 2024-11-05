from alive_progress import alive_bar


class ProgressBarLogic:
    def __init__(self, totalFrames: int):
        """
        Initializes the progress bar for the given range of frames.

        Args:
            totalFrames (int): The total number of frames to process
        """
        self.totalFrames = totalFrames

    def __enter__(self):
        self.bar = alive_bar(
            total=self.totalFrames,
            title="Processing:",
            length=30,
            stats="| {rate} | ETA: {eta}",
            elapsed="Elapsed Time: {elapsed}",
            monitor=" {count}/{total} | [{percent:.0%}] | ",
            unit="frames",
            spinner=None,
        )
        return self.bar.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.bar.__exit__(exc_type, exc_value, traceback)


class ProgressBarDownloadLogic:
    def __init__(self, totalData: int, title: str):
        """
        Initializes the progress bar for the given range of data.

        Args:
            totalData (int): The total amount of data to process
            title (str): The title of the progress bar
        """
        self.totalData = totalData
        self.title = title

    def __enter__(self):
        self.bar = alive_bar(
            total=self.totalData,
            title=self.title,
            length=30,
            stats="| {rate} | ETA: {eta}",
            elapsed="Elapsed Time: {elapsed}",
            monitor=" {count}/{total} | [{percent:.0%}] | ",
            unit="MB",
            spinner=None,
        )
        return self.bar.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.bar.__exit__(exc_type, exc_value, traceback)

    def setTotal(self, newTotal: int):
        """
        Updates the total value of the progress bar.

        Args:
            newTotal (int): The new total value
        """
        self.totalData = newTotal
        self.bar.set_total(newTotal)
