import os

from scenedetect import VideoManager, FrameTimecode, SceneManager
from scenedetect.detectors import AdaptiveDetector
from src.constants import MAINPATH


class AutoClip:
    def __init__(self, input, autoclip_sens, inPoint, outPoint):
        self.input = input
        self.autoclip_sens = autoclip_sens
        self.inPoint = inPoint
        self.outPoint = outPoint

        videoManager = VideoManager([self.input])
        sceneManger = SceneManager()
        sceneManger.add_detector(
            AdaptiveDetector(adaptive_threshold=self.autoclip_sens)
        )

        if self.outPoint != 0:
            startTime = FrameTimecode(self.inPoint, videoManager.get_framerate())
            endTime = FrameTimecode(self.outPoint, videoManager.get_framerate())
            videoManager.set_duration(start_time=startTime, end_time=endTime)

        videoManager.start()
        sceneManger.detect_scenes(frame_source=videoManager, show_progress=True)

        sceneList = sceneManger.get_scene_list()

        with open(os.path.join(MAINPATH, "autoclipresults.txt"), "w") as f:
            for i, scene in enumerate(sceneList):
                startTime = scene[0].get_seconds()
                endTime = scene[1].get_seconds()
                print(f"Scene {i + 1}: Start Time {startTime} - End Time {endTime}")
                f.write(f"{endTime}\n")
