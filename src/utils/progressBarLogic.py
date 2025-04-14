from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
)
import src.constants as cs
from rich.progress import ProgressColumn
from time import time
import os
import json
import logging
import psutil


class FPSColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.startTime = None

    def render(self, task):
        if self.startTime is None:
            self.startTime = time()
        elapsed = time() - self.startTime
        fps = task.completed / elapsed if elapsed > 0 else 0
        return f"FPS: [magenta]{fps:.2f}[/magenta]"


class MemoryColumn(ProgressColumn):
    def __init__(self, totalFrames: int):
        super().__init__()
        self.advanceCount = 0
        self.updateInterval = max(1, totalFrames // 1000)
        self.cachedMem = 0

    def render(self, task):
        if self.advanceCount % self.updateInterval == 0:
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / (1024 * 1024)
            if mem > self.cachedMem:
                self.cachedMem = mem
            else:
                self.cachedMem = (self.cachedMem + mem) / 2
        self.advanceCount += 1
        return f"Mem: [yellow]{self.cachedMem:.1f}MB[/yellow]"


class ProgressBarLogic:
    def __init__(
        self,
        totalFrames: int,
    ):
        """
        Initializes the progress bar for the given range of frames.

        Args:
            totalFrames (int): The total number of frames to process
        """
        self.totalFrames = totalFrames

    def __enter__(self):
        if cs.ADOBE:
            self.advanceCount = 0
            self.updateInterval = max(
                1, self.totalFrames // 100
            )  # 1 update per 1% of total frames
            logging.info(f"Update interval: {self.updateInterval} frames")

            data = {
                "currentFrame": 0,
                "totalFrames": self.totalFrames,
            }
            # Write to a JSON file under a 'logs' folder in the project directory
            self.logFile = os.path.join(cs.MAINPATH, "progressLog.json")
            with open(self.logFile, "w") as f:
                json.dump(data, f, indent=4)
        else:
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
                MemoryColumn(self.totalFrames),
                "•",
                TextColumn("Frames: [green]{task.completed}/{task.total}[/green]"),
            )
            self.task = self.progress.add_task("Processing:", total=self.totalFrames)
            self.progress.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not cs.ADOBE:
            self.progress.stop()

    def advance(self, advance=1):
        if not cs.ADOBE:
            self.progress.update(self.task, advance=advance)

        if cs.ADOBE:
            if not hasattr(self, "_completed"):
                self._completed = 0

            self._completed += advance
            self.advanceCount += 1

            if self.advanceCount % self.updateInterval == 0:
                data = {
                    "currentFrame": self._completed,
                    "totalFrames": self.totalFrames,
                }
                with open(self.logFile, "w") as f:
                    json.dump(data, f, indent=4)

    def __call__(self, advance=1):
        self.advance(advance)

    def updateTotal(self, newTotal: int):
        """
        Updates the total value of the progress bar.

        Args:
            newTotal (int): The new total value
        """
        self.totalFrames = newTotal
        if not cs.ADOBE:
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
            TextColumn("Data: [cyan]{task.completed}/{task.total} MB[/cyan]"),
        )
        self.task = self.progress.add_task(self.title, total=self.totalData)
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
        task = self.progress.tasks[self.task]
        if not hasattr(task, "start_time") or task.start_time is None:
            task.start_time = time()
        self.progress.update(self.task, advance=advance)

    def __call__(self, advance=1):
        self.advance(advance)
