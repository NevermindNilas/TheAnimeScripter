from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
)
from rich.progress import ProgressColumn
from time import time


class FPSColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def render(self, task):
        if self.start_time is None:
            self.start_time = time()
        elapsed = time() - self.start_time
        fps = task.completed / elapsed if elapsed > 0 else 0
        return f"FPS: [magenta]{fps:.2f}[/magenta]"


class ProgressBarLogic:
    def __init__(self, totalFrames: int):
        """
        Initializes the progress bar for the given range of frames.

        Args:
            totalFrames (int): The total number of frames to process
        """
        self.totalFrames = totalFrames

    def __enter__(self):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "•",
            TextColumn("Elapsed Time:"),
            TimeElapsedColumn(),
            "•",
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            "•",
            FPSColumn(),
            "•",
            TextColumn("Frames: [green]{task.completed}/{task.total}[/green]"),
        )
        self.task = self.progress.add_task("Processing:", total=self.totalFrames)
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.stop()

    def advance(self, advance=1):
        self.progress.update(self.task, advance=advance)

    def __call__(self, advance=1):
        self.advance(advance)

    def updateTotal(self, newTotal: int):
        """
        Updates the total value of the progress bar.

        Args:
            newTotal (int): The new total value
        """
        self.totalFrames = newTotal
        self.progress.update(self.task, total=newTotal)


class ProgressBarDownloadLogic:
    def __init__(self, totalData: int, title: str):
        """
        Initializes the progress bar for the given range of data.

        Args:
            totalData (int): The total amount of data to process
            title (str): The title of the progress bar
        """
        self.totalData = totalData - 1
        self.title = title

    def __enter__(self):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "•",
            TextColumn("Elapsed Time:"),
            TimeElapsedColumn(),
            "•",
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            "•",
            TextColumn("Speed: [magenta]{task.fields[mbps]:.2f} MB/s[/magenta]"),
            "•",
            TextColumn("Data: [cyan]{task.completed}/{task.total} MB[/cyan]"),
        )
        self.task = self.progress.add_task(self.title, total=self.totalData, mbps=0.0)
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.stop()

    def setTotal(self, newTotal: int):
        """
        Updates the total value of the progress bar.

        Args:
            newTotal (int): The new total value
        """
        self.totalData = newTotal
        self.progress.update(self.task, total=newTotal)

    def advance(self, advance=1):
        elapsed = time() - self.progress.tasks[self.task].start_time
        mbps = (
            (self.progress.tasks[self.task].completed / elapsed) if elapsed > 0 else 0
        )
        self.progress.update(self.task, advance=advance, mbps=mbps)

    def __call__(self, advance=1):
        self.advance(advance)
