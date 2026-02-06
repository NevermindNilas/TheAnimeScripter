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

progressRefreshPerSec = 10  # Rich refresh frequency
fpsCacheInterval = 0.25  # seconds to cache FPS column output

import os
import logging
import psutil


class SpeedColumn(ProgressColumn):
    """Displays the current download speed in MB/s."""

    def render(self, task):
        elapsed = task.elapsed or 0
        speed = task.completed / elapsed if elapsed > 0 else 0
        return f"Speed: [magenta]{speed:.2f} MB/s[/magenta]"


class FPSColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.startTime = None
        self._lastCacheTime = 0.0
        self._cachedStr = None

    def render(self, task):
        now = time()
        if self.startTime is None:
            self.startTime = now
        if (now - self._lastCacheTime) < fpsCacheInterval and self._cachedStr is not None:
            return self._cachedStr
        elapsed = now - self.startTime
        fps = task.completed / elapsed if elapsed > 0 else 0
        s = f"FPS: [magenta]{fps:.2f}[/magenta]"
        self._cachedStr = s
        self._lastCacheTime = now
        return s


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
        title: str = "Processing",
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
            self.updateInterval = max(10, self.totalFrames // 200)
            logging.info(f"Update interval: {self.updateInterval} frames")

            self.startTime = time()
            self.nextUpdateFrame = self.updateInterval
            self._adobePayload = {
                "currentFrame": 0,
                "totalFrames": self.totalFrames,
                "fps": 0.0,
                "eta": 0.0,
                "elapsedTime": 0.0,
                "status": "Processing...",
            }
            self._adobeUpdate = progressState.update

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
                TextColumn("Frames: [green]{task.completed}/{task.total}[/green]"),
                refresh_per_second=progressRefreshPerSec,
            )
            self.task = self.progress.add_task("Processing:", total=self.totalFrames)
            self.progress.start()
            self._lastRefreshTime = time()
            self._refreshInterval = 1.0 / progressRefreshPerSec
            self._framesSinceRefresh = 0

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

            if self.completed >= getattr(self, "nextUpdateFrame", self.updateInterval) or self.completed >= self.totalFrames:
                currentTime = time()
                elapsedTime = currentTime - self.startTime
                fps_val = self.completed / elapsedTime if elapsedTime > 0 else 0.0

                if fps_val > 0 and self.completed < self.totalFrames:
                    remainingFrames = self.totalFrames - self.completed
                    eta = remainingFrames / fps_val
                else:
                    eta = 0.0

                self._adobePayload["currentFrame"] = self.completed
                self._adobePayload["fps"] = round(fps_val, 2)
                self._adobePayload["eta"] = eta
                self._adobePayload["elapsedTime"] = elapsedTime
                self._adobeUpdate(self._adobePayload)

                self.nextUpdateFrame = self.completed + self.updateInterval

        else:
            self.progress.update(self.task, advance=advance, refresh=False)
            self._framesSinceRefresh += advance
            now = time()
            if (now - self._lastRefreshTime) >= self._refreshInterval:
                self.progress.refresh()
                self._lastRefreshTime = now
                self._framesSinceRefresh = 0

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
            SpeedColumn(),
            "•",
            TextColumn("Data: [cyan]{task.completed}/{task.total} MB[/cyan]"),
            refresh_per_second=progressRefreshPerSec,
        )
        self.task = self.progress.add_task(self.title, total=self.totalData)
        self.progress.start()
        # bookkeeping for throttled refresh and one-time start
        self._started = False
        self._lastRefreshTime = time()
        self._refreshInterval = 1.0 / progressRefreshPerSec
        self._framesSinceRefresh = 0
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
        if not getattr(self, "_started", False):
            try:
                self.progress.tasks[self.task].start_time = time()
            except Exception:
                pass
            self._started = True

        self.progress.update(self.task, advance=advance, refresh=False)
        self._framesSinceRefresh += advance
        now = time()
        if (now - self._lastRefreshTime) >= self._refreshInterval:
            self.progress.refresh()
            self._lastRefreshTime = now
            self._framesSinceRefresh = 0

    def __call__(self, advance=1):
        self.advance(advance)
