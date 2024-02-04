import os

from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector


class Scenechange():
    def __init__(self, input, scenechange_sens, output_dir, inPoint, outPoint):
        self.input = input
        self.scenechange_sens = scenechange_sens
        self.output_dir = output_dir
        self.inPoint = inPoint
        self.outPoint = outPoint

    def run(self):
        video_manager = VideoManager([self.input])
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=self.scenechange_sens))

        if self.outPoint != 0:
            video_manager.set_duration(
                start_time=self.inPoint, end_time=self.outPoint)

        video_manager.set_downscale_factor()

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()

        with open(os.path.join(self.output_dir, 'scenechangeresults.txt'), 'w') as f:
            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                print(
                    f'Scene {i + 1}: Start Time {start_time} - End Time {end_time}')
                f.write(f"{end_time}\n")
