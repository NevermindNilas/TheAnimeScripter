from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
)
from rich.live import Live
from rich.table import Table
import src.constants as cs
from rich.progress import ProgressColumn
from time import time, perf_counter
from src.utils.aeComms import progressState

import os
import logging
import psutil


class EnhancedSpeedColumn(ProgressColumn):
    """Displays the current processing speed with adaptive units."""

    def render(self, task):
        elapsed = task.elapsed or 0
        speed = task.completed / elapsed if elapsed > 0 else 0

        if speed >= 1000:
            return f"[bold cyan]{speed / 1000:.1f}K/s[/bold cyan]"
        else:
            return f"[bold cyan]{speed:.1f}/s[/bold cyan]"


class ModernFPSColumn(ProgressColumn):
    """Enhanced FPS column with better timing and visual styling."""

    def __init__(self):
        super().__init__()
        self.startTime = None
        self.lastUpdate = 0
        self.fpsHistory = []
        self.maxHistory = 10

    def render(self, task):
        if self.startTime is None:
            self.startTime = perf_counter()

        currentTime = perf_counter()
        elapsed = currentTime - self.startTime

        if elapsed > 0:
            currentFps = task.completed / elapsed

            # Smooth FPS calculation using moving average
            self.fpsHistory.append(currentFps)
            if len(self.fpsHistory) > self.maxHistory:
                self.fpsHistory.pop(0)

            avgFps = sum(self.fpsHistory) / len(self.fpsHistory)

            if avgFps >= 100:
                return f"[bold green]{avgFps:.1f} FPS[/bold green]"
            elif avgFps >= 30:
                return f"[bold yellow]{avgFps:.2f} FPS[/bold yellow]"
            else:
                return f"[bold red]{avgFps:.2f} FPS[/bold red]"

        return "[dim]0.0 FPS[/dim]"


class SmartMemoryColumn(ProgressColumn):
    """Optimized memory tracking with better performance and accuracy."""

    def __init__(self, totalFrames: int):
        super().__init__()
        self.totalFrames = totalFrames
        self.updateInterval = max(5, totalFrames // 500)  # Less frequent updates
        self.lastUpdateFrame = 0
        self.cachedMemory = 0
        self.peakMemory = 0
        self.process = psutil.Process(os.getpid())

    def render(self, task):
        framesSinceUpdate = task.completed - self.lastUpdateFrame

        if framesSinceUpdate >= self.updateInterval or task.completed == 0:
            self.lastUpdateFrame = task.completed
            try:
                memory_info = self.process.memory_info()
                currentMemory = memory_info.rss / (1024 * 1024)  # MB

                if currentMemory > self.peakMemory:
                    self.peakMemory = currentMemory

                if self.cachedMemory == 0:
                    self.cachedMemory = currentMemory
                else:
                    self.cachedMemory = 0.8 * self.cachedMemory + 0.2 * currentMemory

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Recreate process handle if needed
                try:
                    self.process = psutil.Process(os.getpid())
                except Exception:
                    pass

        # Color code based on memory usage
        if self.cachedMemory < 500:
            color = "green"
        elif self.cachedMemory < 1000:
            color = "yellow"
        else:
            color = "red"

        return f"[{color}]{self.cachedMemory:.0f}MB[/{color}] [dim](↑{self.peakMemory:.0f})[/dim]"


class EnhancedDownloadColumn(ProgressColumn):
    """Download column that displays in MB instead of bytes."""

    def render(self, task):
        completed_mb = task.completed / (1024 * 1024)
        total_mb = task.total / (1024 * 1024) if task.total else 0

        if task.total:
            return f"[cyan]{completed_mb:.1f}/{total_mb:.1f} MB[/cyan]"
        else:
            return f"[cyan]{completed_mb:.1f} MB[/cyan]"


class EnhancedTransferSpeedColumn(ProgressColumn):
    """Transfer speed column that displays in MB/s instead of bytes/s."""

    def render(self, task):
        if task.speed is None:
            return "[dim]? MB/s[/dim]"

        speed_mb = task.speed / (1024 * 1024)

        if speed_mb >= 1:
            return f"[magenta]{speed_mb:.1f} MB/s[/magenta]"
        else:
            speed_kb = task.speed / 1024
            return f"[magenta]{speed_kb:.1f} KB/s[/magenta]"


class StatusColumn(ProgressColumn):
    """Displays current processing status with visual indicators."""

    def render(self, task):
        if task.completed == 0:
            return "[dim]Initializing...[/dim]"
        elif task.finished:
            return "[bold green]✓ Complete[/bold green]"
        elif task.percentage and task.percentage > 95:
            return "[bold cyan]Finalizing...[/bold cyan]"
        else:
            return "[bold blue]Processing...[/bold blue]"


class StackedProgress:
    """A progress display with title on top and progress bar below."""

    def __init__(self, title: str, total: int, columns: list):
        self.title = title
        self.total = total
        self.columns = columns
        self.progress = None
        self.task = None
        self.live = None

    def __enter__(self):
        # Create progress without description column since we'll show title separately
        progress_columns = [
            col
            for col in self.columns
            if not isinstance(col, TextColumn) or "{task.description}" not in str(col)
        ]

        self.progress = Progress(
            *progress_columns,
            expand=True,
            refresh_per_second=10,
        )

        self.task = self.progress.add_task("", total=self.total)

        # Create layout with title on top
        table = Table.grid()
        table.add_row(f"[bold bright_white]{self.title}[/bold bright_white]")
        table.add_row(self.progress)

        self.live = Live(table, refresh_per_second=10)
        self.live.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.live:
            self.live.stop()

    def update(self, advance=0, **kwargs):
        if self.progress:
            self.progress.update(self.task, advance=advance, **kwargs)

    def advance(self, advance=1):
        self.update(advance=advance)


class ProgressBarLogic:
    def __init__(
        self,
        totalFrames: int,
        title: str = "",
    ):
        """
        Initializes the progress bar for the given range of frames.

        Args:
            totalFrames (int): The total number of frames to process"""
        self.totalFrames = totalFrames
        self.completed = 0
        if title is not None:
            self.title = f"🎬 Working on: {title}"
        else:
            self.title = "🎬 Processing Frames:"

    def __enter__(self):
        if cs.ADOBE:
            # Adobe mode: NO visual progress bar whatsoever!
            # Only internal tracking for API communication
            self.advanceCount = 0
            self.updateInterval = max(10, self.totalFrames // 200)
            self.startTime = time()
            logging.info("Adobe mode: Progress tracking enabled (no visual display)")

        else:
            columns = [
                TextColumn(
                    "[progress.percentage]{task.percentage:>3.1f}%", style="bright_cyan"
                ),
                BarColumn(
                    bar_width=None,
                    complete_style="bold green",
                    finished_style="bold bright_green",
                    pulse_style="bold blue",
                ),
                "│",
                MofNCompleteColumn(),
                "│",
                TimeElapsedColumn(),
                "│",
                TimeRemainingColumn(),
                "│",
                ModernFPSColumn(),
                "│",
                SmartMemoryColumn(self.totalFrames),
                "│",
                StatusColumn(),
                "│",
            ]

            self.stackedProgress = StackedProgress(
                title=self.title, total=self.totalFrames, columns=columns
            )
            self.stackedProgress.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if cs.ADOBE:
            # Adobe mode: Only update internal state, no visual cleanup needed
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
            # Regular mode: Clean up visual progress display
            if hasattr(self, "stackedProgress"):
                self.stackedProgress.__exit__(exc_type, exc_value, traceback)

    def advance(self, advance=1):
        if cs.ADOBE:
            # Adobe mode: Only internal tracking, NO visual progress
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
            # Regular mode: Update visual progress display
            if hasattr(self, "stackedProgress"):
                self.stackedProgress.advance(advance)

    def __call__(self, advance=1):
        self.advance(advance)

    def updateTotal(self, newTotal: int):
        """
        Updates the total value of the progress bar.

        Args:
            newTotal (int): The new total value
        """
        self.totalFrames = newTotal
        if not cs.ADOBE and hasattr(self, "stackedProgress"):
            # Update the total in the stacked progress
            self.stackedProgress.progress.update(
                self.stackedProgress.task, total=newTotal
            )


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
        # Use stacked progress for downloads too
        columns = [
            TextColumn(
                "[progress.percentage]{task.percentage:>3.1f}%", style="bright_cyan"
            ),
            BarColumn(
                bar_width=None,
                complete_style="bold cyan",
                finished_style="bold bright_cyan",
            ),
            "│",
            EnhancedDownloadColumn(),
            "│",
            EnhancedTransferSpeedColumn(),
            "│",
            TimeElapsedColumn(),
            "│",
            TimeRemainingColumn(),
            "│",
        ]

        self.stackedProgress = StackedProgress(
            title=self.title, total=self.totalData, columns=columns
        )
        self.stackedProgress.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "stackedProgress"):
            self.stackedProgress.__exit__(exc_type, exc_value, traceback)

    def setTotal(self, newTotal: int):
        """
        Updates the total value of the progress bar.

        Args:
            newTotal (int): The new total value"""
        self.totalData = newTotal
        if hasattr(self, "stackedProgress"):
            self.stackedProgress.progress.update(
                self.stackedProgress.task, total=newTotal
            )

    def advance(self, advance=1):
        if hasattr(self, "stackedProgress"):
            # Set start time if needed
            task = self.stackedProgress.progress.tasks[self.stackedProgress.task]
            if task.start_time is None:
                task.start_time = time()
            self.stackedProgress.advance(advance)

    def __call__(self, advance=1):
        self.advance(advance)
