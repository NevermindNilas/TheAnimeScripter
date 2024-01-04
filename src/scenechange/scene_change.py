import subprocess
import re
import os

class Scenechange():
    def __init__(self, input, ffmpeg_path, scenechange_sens, output_dir):
        self.input = input
        self.ffmpeg_path = ffmpeg_path
        self.scenechange_sens = scenechange_sens
        self.output_dir = output_dir

    def run(self):
        command = [
            self.ffmpeg_path,
            '-i', self.input,
            '-filter:v', f"select='gt(scene,{self.scenechange_sens})',showinfo",
            '-f', 'null',
            '-'
        ]
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()

        pts_times = re.findall(r'pts_time:(\d+\.\d+)', output)
        
        with open(os.path.join(self.output_dir, 'scenechangeresults.txt'), 'w') as f:
            for pts_time in pts_times:
                f.write(str(float(pts_time)) + '\n')