import os

from scenedetect import VideoManager, FrameTimecode, SceneManager
from scenedetect.detectors import ContentDetector
from src.constants import MAINPATH


class AutoClip:
    def __init__(self, input, autoclip_sens, inPoint, outPoint):
        self.input = input
        self.autoclip_sens = autoclip_sens
        self.inPoint = inPoint
        self.outPoint = outPoint

        video_manager = VideoManager([self.input])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.autoclip_sens))

        if self.outPoint != 0:
            start_time = FrameTimecode(self.inPoint, video_manager.get_framerate())
            end_time = FrameTimecode(self.outPoint, video_manager.get_framerate())
            video_manager.set_duration(start_time=start_time, end_time=end_time)

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)

        scene_list = scene_manager.get_scene_list()

        with open(os.path.join(MAINPATH, "autoclipresults.txt"), "w") as f:
            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                print(f"Scene {i + 1}: Start Time {start_time} - End Time {end_time}")
                f.write(f"{end_time}\n")
