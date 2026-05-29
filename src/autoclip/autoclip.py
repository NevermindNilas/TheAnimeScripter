import os
import logging
import threading
from time import sleep

from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import AdaptiveDetector

import src.constants as cs
from src.utils.progressBarLogic import ProgressBarLogic
from src.utils.logAndPrint import logAndPrint


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

    def __init__(self, input, autoclip_sens, inPoint, outPoint):
        self.input = input
        self.autoclip_sens = autoclip_sens
        self.inPoint = inPoint
        self.outPoint = outPoint

        if cs.ADOBE:
            from src.utils.aeComms import progressState

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

    def _run(self):
        video = open_video(self.input)
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
                endTimecode.get_frames()
                if endTimecode
                else video.duration.get_frames()
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

            with ProgressBarLogic(totalFrames, title="AutoClip (pyscenedetect)") as pbar:
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
