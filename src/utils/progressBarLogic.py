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
from src.utils.aeComms import progressState

import os
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
        self.process = psutil.Process(os.getpid())
        self.lastUpdate = 0

    def render(self, task):
        self.advanceCount += 1
        if self.advanceCount - self.lastUpdate >= self.updateInterval:
            self.lastUpdate = self.advanceCount
            try:
                mem = self.process.memory_info().rss / (1024 * 1024)
                if mem > self.cachedMem:
                    self.cachedMem = mem
                else:
                    self.cachedMem = (self.cachedMem + mem) / 2
            except psutil.NoSuchProcess:
                self.process = psutil.Process(os.getpid())
        return f"Mem: [yellow]{self.cachedMem:.1f}MB[/yellow]"


class ProgressBarLogic:
    def __init__(
        self,
        totalFrames: int,
    ):
        """
        Initializes the progress bar for the given range of frames.

        Args:
            totalFrames (int): The total number of frames to process"""
        self.totalFrames = totalFrames
        self.completed = 0

    def __enter__(self):
        if cs.ADOBE:
            self.advanceCount = 0
            # More frequent updates - every 0.5% or at least every 10 frames
            self.updateInterval = max(10, self.totalFrames // 200)
            logging.info(f"Update interval: {self.updateInterval} frames")

            # Initialize timing for FPS and ETA calculations
            self.startTime = time()

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
        if cs.ADOBE:
            currentTime = time()
            elapsedTime = currentTime - self.startTime
            fps = self.completed / elapsedTime if elapsedTime > 0 else 0

            progressState.update(
                {
                    "currentFrame": self.completed,
                    "totalFrames": self.totalFrames,
                    "fps": round(fps, 2),
                    "eta": 0.0,
                    "elapsedTime": elapsedTime,
                    "status": "Finishing...",
                }
            )
        else:
            self.progress.stop()

    def advance(self, advance=1):
        if cs.ADOBE:
            self.completed += advance
            self.advanceCount += advance

            framesSinceLastUpdate = self.completed % self.updateInterval
            shouldUpdate = (
                framesSinceLastUpdate < advance or self.completed >= self.totalFrames
            )

            if shouldUpdate:
                currentTime = time()
                elapsedTime = currentTime - self.startTime
                fps = self.completed / elapsedTime if elapsedTime > 0 else 0

                if fps > 0 and self.completed < self.totalFrames:
                    remainingFrames = self.totalFrames - self.completed
                    eta = remainingFrames / fps
                else:
                    eta = 0

                progressState.update(
                    {
                        "currentFrame": self.completed,
                        "totalFrames": self.totalFrames,
                        "fps": round(fps, 2),
                        "eta": eta,
                        "elapsedTime": elapsedTime,
                        "status": "Processing...",
                    }
                )

        else:
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
            newTotal (int): The new total value"""
        self.totalData = newTotal
        self.progress.update(self.task, total=newTotal)

    def advance(self, advance=1):
        task = self.progress.tasks[self.task]
        if task.start_time is None:
            task.start_time = time()
        self.progress.update(self.task, advance=advance)

    def __call__(self, advance=1):
        self.advance(advance)
