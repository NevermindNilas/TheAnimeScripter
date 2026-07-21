"""PySceneDetect ``VideoStream`` backed by nelux.

The default autoclip path is ~93% cv2-software-decode-bound; nelux's threaded
decoder reads the same file several times faster, so this adapter feeds
scenedetect's detectors from a ``nelux.VideoReader`` instead of
``VideoStreamCv2``. Frames are delivered as BGR uint8 numpy arrays, matching
what the cv2 backend hands the detectors.

nelux needs torch imported first and the FFmpeg shared DLLs registered
(``src.infra.getFFMPEG.addFfmpegToDllSearchPath``, done during CLI startup).
Callers should treat construction failures as non-fatal and fall back to
``scenedetect.open_video``.
"""

import logging
from fractions import Fraction

import cv2
import numpy as np
from scenedetect import FrameTimecode
from scenedetect.video_stream import VideoStream


class NeluxVideoStream(VideoStream):
    """scenedetect VideoStream over a streaming nelux CPU decode."""

    BACKEND_NAME = "nelux"

    def __init__(self, path: str):
        import torch  # noqa: F401  # this has to be always before nelux!

        # isort: split
        import nelux

        self._nelux = nelux
        self._path = path

        # nelux delay-loads the FFmpeg DLLs at VideoReader construction; if
        # they cannot resolve, the delay-load helper raises SEH 0xC06D007E
        # which kills the process — `except Exception` in the caller never
        # sees it. Preflight the load through ctypes (which honors the same
        # os.add_dll_directory search path) so an unresolvable avcodec turns
        # into a catchable error and the caller can fall back to cv2.
        self._preflightFfmpegDlls()

        # numpy backend returns RGB uint8 HWC ndarrays; force_8bit keeps
        # 10-bit sources at the same depth the cv2 backend would deliver.
        self._reader = self._makeReader()
        props = self._reader
        self._frameRate = float(props.fps or 0.0)
        if self._frameRate <= 0:
            raise ValueError(f"nelux reported no frame rate for {path}")
        self._totalFrames = int(props.total_frames or 0)
        self._size = (int(props.width), int(props.height))
        try:
            self._aspect = float(props.aspect_ratio) or 1.0
        except Exception:
            self._aspect = 1.0

        self._iter = None
        self._frameNumber = 0  # frames decoded so far; last read frame index+1
        self._lastFrame: np.ndarray | None = None
        self._pendingSeek: int | None = None

    def _preflightFfmpegDlls(self) -> None:
        import os

        if os.name != "nt":
            return  # SEH delay-load kills are a Windows mechanism

        import ctypes

        candidates = getattr(self._nelux, "_FFMPEG_DLL_FALLBACKS", None) or {
            "avcodec": ["avcodec-63.dll", "avcodec-62.dll"],
            "avformat": ["avformat-63.dll", "avformat-62.dll"],
        }
        for group in ("avcodec", "avformat"):
            names = candidates.get(group, [])
            if not names:
                continue
            for name in names:
                try:
                    ctypes.WinDLL(name)
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError(
                    f"no loadable {group} DLL among {names}; nelux decode "
                    "would crash the process on construction"
                )

    def _makeReader(self):
        return self._nelux.VideoReader(
            self._path,
            decode_accelerator="cpu",
            backend="numpy",
            force_8bit=True,
        )

    # -- identification ----------------------------------------------------
    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        import os

        return os.path.splitext(os.path.basename(self._path))[0]

    @property
    def is_seekable(self) -> bool:
        return True

    # -- stream properties -------------------------------------------------
    @property
    def frame_rate(self) -> Fraction:
        return Fraction(self._frameRate).limit_denominator(10000)

    @property
    def duration(self) -> FrameTimecode:
        return FrameTimecode(timecode=self._totalFrames, fps=self._frameRate)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._size

    @property
    def aspect_ratio(self) -> float:
        return self._aspect

    @property
    def position(self) -> FrameTimecode:
        return FrameTimecode(
            timecode=max(0, self.frame_number - 1), fps=self._frameRate
        )

    @property
    def position_ms(self) -> float:
        return max(0, self.frame_number - 1) * 1000.0 / self._frameRate

    @property
    def frame_number(self) -> int:
        return self._frameNumber

    # -- decoding ----------------------------------------------------------
    def _ensureIter(self):
        if self._iter is None:
            self._iter = iter(self._reader)
        if self._pendingSeek is not None:
            target = self._pendingSeek
            self._pendingSeek = None
            # Streaming skip: decode-and-discard up to the target frame. This
            # mirrors the trim behavior of the rest of the pipeline (nelux
            # 0.15.x set_range proved unreliable at deep offsets), and a
            # nelux skip still outruns a full cv2 decode.
            while self._frameNumber < target:
                try:
                    next(self._iter)
                except StopIteration:
                    break
                self._frameNumber += 1

    def read(self, decode: bool = True, advance: bool = True):
        self._ensureIter()
        if not advance:
            return self._lastFrame if (decode and self._lastFrame is not None) else True
        try:
            frame = next(self._iter)
        except StopIteration:
            return False
        self._frameNumber += 1
        if not decode:
            return True
        # nelux numpy backend: RGB uint8 HWC -> BGR contiguous for detectors.
        self._lastFrame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        return self._lastFrame

    def reset(self) -> None:
        self._closeIter()
        self._reader = self._makeReader()
        self._frameNumber = 0
        self._lastFrame = None
        self._pendingSeek = None

    def seek(self, target) -> None:
        if isinstance(target, int):
            targetFrame = target
        elif isinstance(target, float):
            targetFrame = int(target * self._frameRate)
        else:
            # FrameTimecode or any TimecodeLike (e.g. "00:01:30" strings).
            targetFrame = FrameTimecode(
                timecode=target, fps=self._frameRate
            ).get_frames()
        if targetFrame < 0:
            raise ValueError(f"invalid seek target: {target}")
        if targetFrame < self._frameNumber:
            self.reset()
        self._pendingSeek = targetFrame

    def _closeIter(self):
        self._iter = None
        try:
            del self._reader
        except Exception as e:
            logging.debug(f"nelux reader teardown: {e}")
