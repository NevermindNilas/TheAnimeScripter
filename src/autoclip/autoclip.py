import json
import logging
import os
import subprocess
import threading
from time import sleep

from scenedetect import FrameTimecode, SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector

import src.constants as cs
from src.infra.logAndPrint import logAndPrint
from src.infra.progressBarLogic import ProgressBarLogic


class AutoClip:
    """
    PySceneDetect-backed scene-change autoclip.

    Default path uses scenedetect's built-in tqdm progress bar
    (``show_progress=True``). When ``cs.ADOBE`` is set, switches to a
    threaded layout: ``detect_scenes`` runs in a worker, main thread polls
    ``video.frame_number`` every 100ms and drives ``ProgressBarLogic``,
    which emits per-frame socket.io events to the Adobe extension.
    """

    POLL_INTERVAL = 0.1

    # Switch the frame source from cv2 to nelux only when the decode saving
    # can pay for importing torch (nelux requires it). Measured on a 3090:
    # nelux decodes ~4.6x faster than cv2 (saves ~2.1ms/frame at 1080p,
    # scaling with pixel count) while the torch import costs ~1.4s, which
    # works out to ~7e8 pixel-frames at break-even. Short or low-res inputs
    # stay on cv2.
    NELUX_PIXEL_FRAMES_THRESHOLD = 7e8

    def __init__(self, input, autoclip_sens, inPoint, outPoint):
        self.input = input
        self.autoclip_sens = autoclip_sens
        self.inPoint = inPoint
        self.outPoint = outPoint

        if cs.ADOBE:
            from src.server.aeComms import progressState

            progressState.update(
                {"status": "Detecting scene changes (pyscenedetect)..."}
            )
            try:
                self._run()
            except Exception as e:
                progressState.setFailed(error=str(e))
                raise
            outPath = os.path.join(cs.WHEREAMIRUNFROM, "autoclipresults.txt")
            progressState.setCompleted(outputPath=outPath)
        else:
            self._run()

    def _isConstantFrameRate(self) -> bool:
        """ffprobe CFR check: r_frame_rate must equal avg_frame_rate.

        The nelux stream derives cut timestamps from frame_number / average
        fps, which is only truthful on constant-frame-rate containers; the
        cv2 backend tracks real PTS, so VFR sources (screen/phone recordings,
        many web downloads) must stay on cv2 or cut seconds drift by the
        accumulated PTS deviation.

        Any probe failure (ffprobe error, non-JSON output, missing stream)
        is treated as "not provably CFR" and returns False, keeping the run
        on cv2 rather than raising.
        """
        try:
            result = subprocess.run(
                [
                    cs.FFPROBEPATH,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=r_frame_rate,avg_frame_rate",
                    "-of",
                    "json",
                    self.input,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            stream = json.loads(result.stdout)["streams"][0]

            def rate(expr: str) -> float:
                num, _, den = expr.partition("/")
                return float(num) / float(den) if den and float(den) else 0.0

            r, avg = rate(stream["r_frame_rate"]), rate(stream["avg_frame_rate"])
            return r > 0 and abs(r - avg) < 0.01
        except Exception as e:
            logging.info(f"CFR probe failed ({e}); treating input as VFR (cv2)")
            return False

    def _neluxWorthIt(self, video) -> bool:
        """True when the nelux frame source should replace cv2 for this run."""
        if self.inPoint:
            # The adapter's seek() is decode-and-discard from frame 0; cv2
            # seeks the container. Deep in-points would decode MORE, not less.
            return False
        width, height = video.frame_size
        totalFrames = video.duration.get_frames()
        if totalFrames * width * height <= self.NELUX_PIXEL_FRAMES_THRESHOLD:
            return False
        return self._isConstantFrameRate()

    def _detect(self, video):
        """Run AdaptiveDetector over ``video``; returns the SceneManager."""
        sceneManager = SceneManager()
        sceneManager.add_detector(
            AdaptiveDetector(adaptive_threshold=self.autoclip_sens)
        )

        fps = video.frame_rate
        startTimecode = FrameTimecode(self.inPoint, fps) if self.inPoint else None
        endTimecode = FrameTimecode(self.outPoint, fps) if self.outPoint else None

        if startTimecode is not None:
            video.seek(startTimecode)

        if cs.ADOBE:
            startFrame = startTimecode.get_frames() if startTimecode else 0
            endFrame = (
                endTimecode.get_frames() if endTimecode else video.duration.get_frames()
            )
            totalFrames = max(1, endFrame - startFrame)

            result = {"exception": None}

            def worker():
                try:
                    sceneManager.detect_scenes(
                        video=video,
                        end_time=endTimecode,
                        show_progress=False,
                    )
                except BaseException as e:
                    result["exception"] = e

            thread = threading.Thread(
                target=worker, name="autoclip-detect", daemon=True
            )
            thread.start()

            with ProgressBarLogic(
                totalFrames, title="AutoClip (pyscenedetect)"
            ) as pbar:
                lastSeen = startFrame
                while thread.is_alive():
                    sleep(self.POLL_INTERVAL)
                    current = video.frame_number
                    delta = current - lastSeen
                    if delta > 0:
                        pbar.advance(min(delta, totalFrames - (lastSeen - startFrame)))
                        lastSeen = current
                thread.join()
                current = video.frame_number
                delta = current - lastSeen
                if delta > 0:
                    pbar.advance(min(delta, totalFrames - (lastSeen - startFrame)))

            if result["exception"] is not None:
                raise result["exception"]
        else:
            sceneManager.detect_scenes(
                video=video,
                end_time=endTimecode,
                show_progress=True,
            )

        return sceneManager

    def _expectedEndFrame(self, video) -> int:
        if self.outPoint:
            return FrameTimecode(self.outPoint, video.frame_rate).get_frames()
        return video.duration.get_frames()

    def _run(self):
        # The default autoclip run is ~93% cv2-decode-bound; for large CFR
        # inputs swap in the nelux-backed stream (several times faster
        # decode), keeping cv2 as the fallback if nelux cannot handle the
        # source.
        video = open_video(self.input)
        usingNelux = False
        try:
            if self._neluxWorthIt(video):
                from src.autoclip.neluxStream import NeluxVideoStream

                video = NeluxVideoStream(self.input)
                usingNelux = True
                logging.info("autoclip: using nelux frame source")
        except Exception as e:
            logging.info(f"nelux autoclip source unavailable ({e}); using cv2")
            video = open_video(self.input)
            usingNelux = False

        sceneManager = self._detect(video)

        # nelux surfaces mid-stream decode errors as a clean end-of-stream,
        # which would silently drop every cut after the damage. If the scan
        # ended early, redo the detection on the decode-error-tolerant cv2
        # backend.
        if usingNelux and video.frame_number < self._expectedEndFrame(video) - 2:
            logging.warning(
                f"nelux autoclip scan ended early at frame {video.frame_number} "
                f"of {self._expectedEndFrame(video)}; re-running with cv2"
            )
            video = open_video(self.input)
            sceneManager = self._detect(video)

        sceneList = sceneManager.get_scene_list()
        outPath = os.path.join(cs.WHEREAMIRUNFROM, "autoclipresults.txt")
        # The last scene's end is the video EOF, not a real cut, so drop it to
        # avoid emitting a spurious boundary at the very end of the video
        # (matching the TransNetV2/Maxxvit behavior).
        cuts = sceneList[:-1]
        with open(outPath, "w") as f:
            for i, scene in enumerate(cuts):
                endSec = scene[1].get_seconds()
                logging.info(f"Scene {i + 1}: cut at {endSec:.3f}s")
                f.write(f"{endSec}\n")
        logAndPrint(f"AutoClip wrote {len(cuts)} cuts to {outPath}", "green")
