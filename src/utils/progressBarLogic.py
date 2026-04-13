from barflow import Progress
from barflow.columns import (
    BarColumn,
    TextColumn,
    DescriptionColumn,
    PercentColumn,
    ElapsedColumn,
    EtaColumn,
    CountColumn,
    CallbackColumn,
)
import src.constants as cs
from time import time
from random import choice
from src.utils.aeComms import progressState

import logging

TITLES = [
    "Handling",
    "Processing",
    "Inferencing",
    "Managing",
    "Dealing With",
    "Working",
    "Executing",
    "Computing",
    "Refining",
    "Producing",
    "Digesting",
]

progressRefreshPerSec = 10
_minInterval = 1.0 / progressRefreshPerSec

_SEP = TextColumn(" \u2502 ", style="bright_black")


def _fpsRender(task):
    elapsed = task.elapsed
    fps = task.completed / elapsed if elapsed > 0 else 0.0
    return f"FPS {fps:6.2f}"


def _byteSpeedRender(task):
    elapsed = task.elapsed
    mbps = (task.completed / elapsed / (1024 * 1024)) if elapsed > 0 else 0.0
    return f"{mbps:6.2f} MB/s"


def _byteCountRender(task):
    mb = 1024 * 1024
    return f"{task.completed / mb:6.2f}/{task.total / mb:.2f} MB"


def _frameColumns(desc: str):
    return (
        DescriptionColumn(style="bold bright_cyan"),
        TextColumn(" "),
        BarColumn(width=None, style="bright_cyan", glyphs="smooth"),
        TextColumn(" "),
        PercentColumn(style="bold white"),
        _SEP,
        ElapsedColumn(style="yellow"),
        TextColumn("<", style="bright_black"),
        EtaColumn(style="bright_yellow"),
        _SEP,
        CallbackColumn(_fpsRender, style="magenta"),
        _SEP,
        CountColumn(style="green"),
    )


def _byteColumns():
    return (
        DescriptionColumn(style="bold bright_cyan"),
        TextColumn(" "),
        BarColumn(width=None, style="bright_cyan", glyphs="smooth"),
        TextColumn(" "),
        PercentColumn(style="bold white"),
        _SEP,
        ElapsedColumn(style="yellow"),
        TextColumn("<", style="bright_black"),
        EtaColumn(style="bright_yellow"),
        _SEP,
        CallbackColumn(_byteSpeedRender, style="magenta"),
        _SEP,
        CallbackColumn(_byteCountRender, style="cyan"),
    )


class ProgressBarLogic:
    def __init__(
        self,
        totalFrames: int,
        title: str = None,
    ):
        """
        Initializes the progress bar for the given range of frames.

        Args:
            totalFrames (int): The total number of frames to process
            title (str): Description shown at the head of the bar
        """
        self.totalFrames = totalFrames
        self.title = title or choice(TITLES)
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
                *_frameColumns(self.title),
                total=self.totalFrames,
                desc=self.title,
                min_interval=_minInterval,
            )
            self.progress.__enter__()

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
            self.progress.__exit__(exc_type, exc_value, traceback)

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
            self.progress.advance(advance)

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
            self.progress.set_total(0, newTotal)


class ProgressBarDownloadLogic:
    def __init__(self, totalData: int, title: str):
        """
        Initializes the progress bar for the given range of data.

        Args:
            totalData (int): Total bytes to download
            title (str): The title of the progress bar
        """
        self.totalData = max(1, int(totalData))
        self.title = title

    def __enter__(self):
        self.progress = Progress(
            *_byteColumns(),
            total=self.totalData,
            desc=self.title,
            min_interval=_minInterval,
        )
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.__exit__(exc_type, exc_value, traceback)

    def setTotal(self, newTotal: int):
        """
        Updates the total value of the progress bar.

        Args:
            newTotal (int): The new total value"""
        self.totalData = newTotal
        self.progress.set_total(0, newTotal)

    def advance(self, advance=1):
        self.progress.advance(advance)

    def __call__(self, advance=1):
        self.advance(advance)
