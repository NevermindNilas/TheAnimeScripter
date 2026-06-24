import logging
import os
from random import choice
from time import time

from barflow import Progress
from barflow.columns import (
    BarColumn,
    CallbackColumn,
    CountColumn,
    DescriptionColumn,
    ElapsedColumn,
    EtaColumn,
    PercentColumn,
    SpinnerColumn,
    TextColumn,
)

import src.constants as cs
from src.server.aeComms import progressState

TITLES = [
    "Handling",
    "Processing",
    "Inferencing",
    "Dealing With",
    "Working",
    "Executing",
    "Computing",
    "Refining",
    "Producing",
    "Digesting",
    "Transforming",
    "Analyzing",
    "Calculating",
    "Rendering",
    "Synthesizing",
    "Optimizing",
    "Finalizing",
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


def _outputInfoRender(outputPath, videoFps=None):
    """Callback factory: current output filesize + estimated bitrate.

    Runs on the render thread at the bar's refresh rate (10 Hz) — one
    os.path.getsize per render, no cost on the frame loop. Returns ""
    until the encoder creates the file.

    With `videoFps`, bitrate is the encoded content's average bitrate
    (size over `completed / videoFps` seconds of output video). Without
    it, falls back to write throughput (size over wall elapsed), which
    says how fast bytes hit the disk, not how heavy the video is.
    """

    def render(task):
        try:
            size = os.path.getsize(outputPath)
        except OSError:
            # Encoder hasn't created the file yet — render nothing,
            # including the separator (it lives here, not in the column
            # list, so no dangling "│" before the first write).
            return ""
        if videoFps and task.completed > 0:
            duration = task.completed / videoFps
        else:
            duration = task.elapsed
        mbps = (size * 8 / duration / 1e6) if duration > 0 else 0.0
        if size >= 1024**3:
            sz = f"{size / 1024**3:.2f} GB"
        else:
            sz = f"{size / (1024 * 1024):.1f} MB"
        return f" │ {sz} ~{mbps:.1f}Mbps"

    return render


def _frameColumns(outputPath=None, videoFps=None):
    cols = [
        SpinnerColumn(name="dots", style="bold bright_cyan"),
        TextColumn(" "),
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
    ]
    if outputPath:
        cols.append(
            CallbackColumn(_outputInfoRender(outputPath, videoFps), style="cyan")
        )
    return tuple(cols)


def _byteColumns():
    return (
        SpinnerColumn(name="dots", style="bold bright_cyan"),
        TextColumn(" "),
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
        outputPath: str = None,
        videoFps: float = None,
    ):
        """
        Initializes the progress bar for the given range of frames.

        Args:
            totalFrames (int): The total number of frames to process
            title (str): Description shown at the head of the bar
            outputPath (str): When set, the bar appends a live
                "filesize ~bitrate" column polled from this file
            videoFps (float): Output video fps; when set, the bitrate is
                the encoded content's average (size / output seconds)
                instead of raw write throughput
        """
        self.totalFrames = totalFrames
        self.title = title or choice(TITLES)
        self.outputPath = outputPath
        self.videoFps = videoFps
        self.completed = 0

    def __enter__(self):
        if cs.ADOBE:
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
                *_frameColumns(self.outputPath, self.videoFps),
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

            if (
                self.completed >= getattr(self, "nextUpdateFrame", self.updateInterval)
                or self.completed >= self.totalFrames
            ):
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

    def advance(self, advance=1):
        self.progress.advance(advance)

    def __call__(self, advance=1):
        self.advance(advance)
