import logging
import os
import shutil
import sys
import threading
from math import exp
from time import monotonic, time

from barflow import Progress
from barflow.columns import BarColumn, CallbackColumn, SpinnerColumn, TextColumn

import src.constants as cs
from src.server.aeComms import progressState

progressRefreshPerSec = 10
_minInterval = 1.0 / progressRefreshPerSec

# Speed and ETA follow the last few seconds of work instead of the whole-run
# average, so warm-up frames (CUDA-graph capture, TRT/ORT first runs) stop
# dragging the ETA and a mid-run slowdown shows up. The terminal bar, the
# plain-text fallback and the After Effects payload all use `_RateMeter`;
# barflow's own `smoothing=` is per-render, so its time constant would drift
# with however often the render thread actually gets to draw.
_RATE_TAU_S = 2.0

# The encoder runs behind the bar (32-deep writer queue plus encoder lookahead),
# so size / frames-counted reads low until the pipeline has filled.
_BITRATE_WARMUP_S = 3.0

# Without a terminal the bar is suppressed and a plain status line is printed
# at this interval instead, so redirected/batch runs still log progress.
_PLAIN_INTERVAL_S = 10.0

# Bars currently open, so a force-exit can end them first. TAS's Ctrl+C
# handler leaves via os._exit, which skips every __exit__: barflow (0.6+)
# hides the cursor while a bar is drawing, so an un-closed bar left the shell
# with no cursor and the interrupt warning printed over the bar line.
_liveBars = set()
_liveLock = threading.Lock()


def _register(bar):
    with _liveLock:
        _liveBars.add(bar)


def _unregister(bar):
    with _liveLock:
        _liveBars.discard(bar)


def closeLiveBars() -> None:
    """End every open bar now, from any thread: barflow restores the cursor and
    finishes the bar line. For force-exit paths only; a bar's own worker may
    keep calling advance() afterwards, which is then a no-op on screen."""
    with _liveLock:
        bars = list(_liveBars)
        _liveBars.clear()
    for bar in bars:
        try:
            bar._abort()
        except Exception:
            pass


def _interactive() -> bool:
    try:
        return sys.stderr is not None and sys.stderr.isatty()
    except Exception:
        return False


def _formatSeconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _formatSize(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    return f"{size / (1024 * 1024):.1f} MB"


class _RateMeter:
    """Items/second as an exponential moving average with `_RATE_TAU_S`."""

    def __init__(self, completed: int = 0, now: float | None = None):
        self.rate = 0.0
        self._primed = False
        self._lastCompleted = completed
        self._lastTime = monotonic() if now is None else now

    def update(self, completed: int, now: float) -> float:
        dt = now - self._lastTime
        if dt <= 0 or (not self._primed and completed == self._lastCompleted):
            # Nothing done yet: priming on 0 would make the meter climb from
            # zero for several time constants. Keep the anchor so the first
            # real sample measures start -> first progress.
            return self.rate
        inst = (completed - self._lastCompleted) / dt
        if self._primed:
            self.rate += (1.0 - exp(-dt / _RATE_TAU_S)) * (inst - self.rate)
        else:
            self.rate = inst
            self._primed = True
        self._lastCompleted = completed
        self._lastTime = now
        return self.rate


class _LiveRate:
    """`_RateMeter` sampled from the render thread, once per drawn frame, and
    shared by the speed and ETA columns. A finished bar reports the whole-run
    average, which is the number worth keeping on screen afterwards."""

    def __init__(self):
        self._meter = _RateMeter()
        self._tick = None

    def __call__(self, task) -> float:
        if task.total and task.completed >= task.total:
            return task.completed / task.elapsed if task.elapsed > 0 else 0.0
        if task.frame_tick != self._tick:
            self._tick = task.frame_tick
            self._meter.update(task.completed, monotonic())
        return self._meter.rate


# Every non-bar field is composed here rather than by barflow's own columns so
# the line can be fitted to the terminal first. barflow's overflow handling
# hides the *bar* before it drops anything; fitting here sheds the least
# useful fields first and keeps the bar.
_MIN_BAR = 20  # bar body cells kept; fields drop before the bar shrinks past it
# The title's name part (file name) is capped at a quarter of the terminal,
# but never below this, before any field is dropped for it.
_MIN_NAME = 16
# On a narrow terminal the name keeps at least this many characters; the
# resolution/fps detail is dropped before the name shrinks past it.
_NAME_FLOOR = 8
# Once every optional field is gone, the bar gives up this many more cells
# before the title is truncated (narrow terminals).
_BAR_SLACK = 10
_SEP = " \u2502 "
# spinner + gap, gap before the bar, bar borders, minimum body, and barflow's
# one-cell safety margin.
_FIXED_CELLS = 2 + 1 + 2 + _MIN_BAR + 1

_RESET = "\x1b[0m"
_C_TITLE = "\x1b[1;96m"
_C_PCT = "\x1b[1;97m"
_C_ELAPSED = "\x1b[33m"
_C_DIM = "\x1b[90m"
_C_ETA = "\x1b[93m"
_C_RATE = "\x1b[35m"
_C_COUNT = "\x1b[32m"
_C_SIZE = "\x1b[36m"


def _termWidth() -> int:
    """Columns of the terminal barflow draws on (stderr)."""
    try:
        return os.get_terminal_size(sys.stderr.fileno()).columns
    except AttributeError, OSError, ValueError:
        return shutil.get_terminal_size().columns


def _paint(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}" if text else ""


def _capName(title, keep, cap):
    """Shorten the name part (everything before the last `keep` characters)
    to `cap` cells, so a long file name cannot push every field off the line."""
    head, tail = title[: len(title) - keep], title[len(title) - keep :]
    if len(head) <= cap:
        return title
    return head[: max(0, cap - 1)].rstrip() + "…" + tail


def _fit(budget, title, fields, dropOrder, keep=0):
    """Choose a variant per optional field so title + fields fit `budget`.

    `fields` maps name -> list of variants, fullest first; "" hides the field.
    Each step of `dropOrder` moves that field to its next variant until the
    line fits; past the last step the bar gives up `_BAR_SLACK` cells, then the
    title is truncated -- ahead of its last `keep` characters, so a detail
    suffix like " │ 2160p/60fps" survives a long file name.
    Returns ({name: chosen text}, title).
    """
    level = dict.fromkeys(fields, 0)

    def used():
        total = len(title)
        for name, variants in fields.items():
            text = variants[level[name]]
            if text:
                total += len(_SEP) + len(text)
        return total

    taken = []
    for name in dropOrder:
        if used() <= budget:
            break
        if level[name] < len(fields[name]) - 1:
            level[name] += 1
            taken.append(name)
    # The last drop can free more than was needed; give back, latest first,
    # any earlier drop that still fits (e.g. the file size once the long
    # counters are gone).
    for name in reversed(taken):
        level[name] -= 1
        if used() > budget:
            level[name] += 1
    over = used() - budget - _BAR_SLACK
    if over > 0:
        head, tail = title[: len(title) - keep], title[len(title) - keep :]
        if len(head) - over - 1 >= min(len(head), _NAME_FLOOR):
            title = head[: len(head) - over - 1] + "\u2026" + tail
        else:
            # Shortening the name further would leave it unrecognisable; the
            # name says which file is running, so the detail goes first --
            # anything still over would make barflow hide the bar.
            over -= len(tail)
            title = (
                head if over <= 0 else head[: max(0, len(head) - over - 1)] + "\u2026"
            )
    return {name: fields[name][level[name]] for name in fields}, title


class _Line:
    """Measures the terminal, fits the line and paints the title and the
    after-bar tail once per drawn frame; the two callbacks share the plan.

    `build(task)` returns (headPlain, headPainted, fields, colors): the head
    (percent and timings) is always shown, `fields` are droppable variants.
    """

    def __init__(self, titleFn, build, dropOrder):
        self._titleFn = titleFn
        self._build = build
        self._dropOrder = dropOrder
        self._tick = None
        self._title = ""
        self._tail = ""

    def _plan(self, task):
        if task.frame_tick == self._tick:
            return
        self._tick = task.frame_tick
        headPlain, headPainted, fields, colors = self._build(task)
        width = _termWidth()
        budget = width - _FIXED_CELLS - len(headPlain)
        title, keep = self._titleFn()
        suffix = title[len(title) - keep :] if keep else ""
        title = _capName(title, keep, max(_MIN_NAME, width // 4))
        chosen, title = _fit(budget, title, fields, self._dropOrder, keep)
        if suffix and title.endswith(suffix):
            # Name in the title colour, the detail styled like the other
            # fields; a detail cut on a very narrow terminal stays plain.
            self._title = (
                _paint(_C_TITLE, title[: -len(suffix)])
                + _paint(_C_DIM, _SEP)
                + _paint(_C_PCT, suffix[len(_SEP) :])
            )
        else:
            self._title = _paint(_C_TITLE, title)
        parts = [headPainted]
        for name, text in chosen.items():
            if text:
                parts.append(_paint(_C_DIM, _SEP) + _paint(colors[name], text))
        self._tail = "".join(parts)

    def title(self, task):
        self._plan(task)
        return self._title

    def tail(self, task):
        self._plan(task)
        return self._tail

    def columns(self):
        return (
            SpinnerColumn(name="dots", style="bold bright_cyan"),
            TextColumn(" "),
            CallbackColumn(self.title),
            TextColumn(" "),
            BarColumn(width=None, style="bright_cyan", glyphs="smooth"),
            CallbackColumn(self.tail),
        )


def _eta(task, rate):
    if task.total and task.completed >= task.total:
        return "00:00"
    if rate <= 0 or not task.total:
        return "--:--"
    return _formatSeconds((task.total - task.completed) / rate)


def _head(task, rate):
    """Percent and elapsed<ETA, in plain (for measuring) and painted form."""
    pct = f"{int(100 * task.completed / task.total) if task.total else 0:3d}%"
    elapsed = _formatSeconds(task.elapsed)
    eta = _eta(task, rate)
    plain = f" {pct}{_SEP}{elapsed}<{eta}"
    painted = (
        " "
        + _paint(_C_PCT, pct)
        + _paint(_C_DIM, _SEP)
        + _paint(_C_ELAPSED, elapsed)
        + _paint(_C_DIM, "<")
        + _paint(_C_ETA, eta)
    )
    return plain, painted


def _fpsVariants(rate, outputPerSource):
    """The bar counts output frames; with interpolation that is not the rate
    the pipeline consumes source frames at, so show both while room allows."""
    short = f"{rate:.1f} fps"
    if outputPerSource == 1:
        return [short]
    return [f"{short} \u00b7 {rate / outputPerSource:.1f} src", short]


def _sizeVariants(outputPath, videoFps, task):
    """Current output filesize + estimated bitrate, polled at render rate.

    Nothing until the encoder creates the file; the bitrate is held back for
    the first `_BITRATE_WARMUP_S` while the encoder is still behind the bar.
    With `videoFps`, bitrate is the encoded content's average (size over
    `completed / videoFps` seconds of output video); without it, write
    throughput (size over wall elapsed).
    """
    if not outputPath:
        return [""]
    try:
        size = os.path.getsize(outputPath)
    except OSError:
        return [""]
    sizeText = _formatSize(size)
    if task.elapsed < _BITRATE_WARMUP_S:
        return [sizeText, ""]
    if videoFps and task.completed > 0:
        duration = task.completed / videoFps
    else:
        duration = task.elapsed
    mbps = (size * 8 / duration / 1e6) if duration > 0 else 0.0
    return [f"{sizeText} ~{mbps:.1f}Mbps", sizeText, ""]


# Shed first -> last: the count repeats the percentage; bitrate and source fps
# are refinements; the dedup/cut counters outlive the file size.
_FRAME_DROP_ORDER = ("count", "size", "fps", "stats", "size", "stats")
_FRAME_COLORS = {"fps": _C_RATE, "count": _C_COUNT, "size": _C_SIZE, "stats": _C_DIM}


def _frameLine(bar):
    rate = _LiveRate()

    def build(task):
        r = rate(task)
        plain, painted = _head(task, r)
        statsLong, statsShort = bar._stats
        fields = {
            "fps": _fpsVariants(r, bar.outputPerSource),
            "count": [f"{task.completed}/{task.total}", ""],
            "size": _sizeVariants(bar.outputPath, bar.videoFps, task),
            "stats": [statsLong, statsShort or "", ""] if statsLong else [""],
        }
        return plain, painted, fields, _FRAME_COLORS

    return _Line(lambda: (bar.title, bar._titleKeep), build, _FRAME_DROP_ORDER)


_BYTE_COLORS = {"speed": _C_RATE, "count": _C_SIZE}


def _byteLine(title):
    rate = _LiveRate()
    mb = 1024 * 1024

    def build(task):
        r = rate(task)
        plain, painted = _head(task, r)
        fields = {
            "speed": [f"{r / mb:.2f} MB/s"],
            "count": [f"{task.completed / mb:.2f}/{task.total / mb:.2f} MB", ""],
        }
        return plain, painted, fields, _BYTE_COLORS

    return _Line(lambda: (title, 0), build, ("count",))


class ProgressBarLogic:
    def __init__(
        self,
        totalFrames: int,
        title: str = None,
        outputPath: str = None,
        videoFps: float = None,
        outputPerSource: float = 1,
        titleDetail: str = None,
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
            outputPerSource (float): Output frames counted per source frame
                (the interpolation factor); when not 1 the speed column
                shows both output and source fps
            titleDetail (str): Shown after the title as "title │ detail";
                on a narrow terminal the title is shortened, never the detail
        """
        self.totalFrames = totalFrames
        self.title = title or "Processing"
        self._titleKeep = 0
        if titleDetail:
            suffix = f"{_SEP}{titleDetail}"
            self.title += suffix
            self._titleKeep = len(suffix)
        self.outputPath = outputPath
        self.videoFps = videoFps
        self.outputPerSource = outputPerSource or 1
        self.completed = 0
        self._stats = ("", "")

    def __enter__(self):
        self._interactive = False
        if cs.ADOBE:
            self.updateInterval = max(10, self.totalFrames // 200)
            logging.info(f"Update interval: {self.updateInterval} frames")

            self.startTime = time()
            self._rate = _RateMeter(0, monotonic())
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
            return self

        self._interactive = _interactive()
        if not self._interactive:
            self.startTime = monotonic()
            self._rate = _RateMeter(0, self.startTime)
            self._nextPlainLine = self.startTime + _PLAIN_INTERVAL_S

        self.progress = Progress(
            *_frameLine(self).columns(),
            total=self.totalFrames,
            desc=self.title,
            min_interval=_minInterval,
            disable=not self._interactive,
        )
        self.progress.__enter__()
        self._aborted = False
        _register(self)
        return self

    def _abort(self):
        """closeLiveBars(): end the bar from another thread before a force-exit."""
        self._aborted = True
        self.progress.close()
        if not self._interactive:
            self._printPlainLine(monotonic(), final=True)

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
            return
        _unregister(self)
        if self._aborted:
            return  # closeLiveBars() already ended it
        self.progress.__exit__(exc_type, exc_value, traceback)
        if not self._interactive:
            self._printPlainLine(monotonic(), final=True)

    def advance(self, advance=1):
        self.completed += advance
        if cs.ADOBE:
            if (
                self.completed >= getattr(self, "nextUpdateFrame", self.updateInterval)
                or self.completed >= self.totalFrames
            ):
                elapsedTime = time() - self.startTime
                fps_val = self._rate.update(self.completed, monotonic())

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
            return

        self.progress.advance(advance)
        if not self._interactive and not self._aborted:
            now = monotonic()
            if now >= self._nextPlainLine:
                self._printPlainLine(now)
                self._nextPlainLine = now + _PLAIN_INTERVAL_S

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

    def setTitle(self, title: str):
        """Rename the bar mid-run, e.g. to name the current pass. The title
        column reads `self.title` on every render."""
        self.title = title
        self._titleKeep = 0

    def setStats(self, text: str, short: str = None):
        """Live pipeline counters shown after the bar (e.g. dedup drops), with
        an optional `short` form used when the terminal is too narrow for
        `text`. Read by the render thread; a plain attribute store."""
        self._stats = (text, short)

    def _printPlainLine(self, now: float, final: bool = False):
        elapsed = now - self.startTime
        rate = self._rate.update(self.completed, now)
        if final:
            # The whole-run average is the meaningful number for a summary.
            rate = self.completed / elapsed if elapsed > 0 else 0.0
        total = self.totalFrames
        pct = f"{100 * self.completed / total:.0f}%" if total else "?"
        line = f"{self.title}: {self.completed}/{total} ({pct}) {rate:.2f} fps"
        if self.outputPerSource != 1:
            line += f" ({rate / self.outputPerSource:.2f} src)"
        line += f", elapsed {_formatSeconds(elapsed)}"
        if not final and rate > 0 and total > self.completed:
            line += f", ETA {_formatSeconds((total - self.completed) / rate)}"
        if self._stats[0]:
            line += f", {self._stats[0]}"
        print(line, file=sys.stderr, flush=True)


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
            *_byteLine(self.title).columns(),
            total=self.totalData,
            desc=self.title,
            min_interval=_minInterval,
            disable=not _interactive(),
        )
        self.progress.__enter__()
        _register(self)
        return self

    def _abort(self):
        self.progress.close()

    def __exit__(self, exc_type, exc_value, traceback):
        _unregister(self)
        self.progress.__exit__(exc_type, exc_value, traceback)

    def advance(self, advance=1):
        self.progress.advance(advance)

    def __call__(self, advance=1):
        self.advance(advance)
