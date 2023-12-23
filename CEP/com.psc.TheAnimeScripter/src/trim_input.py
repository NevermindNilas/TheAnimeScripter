import os
import subprocess
class trim_input_dedup():
    def __init__(self, input, output, inpoint, outpoint, mpdecimate_params, ffmpeg_path):
        self.input = input
        self.output = output
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.mpdecimate_params = mpdecimate_params
        self.ffmpeg_path = ffmpeg_path
            
    def run(self):        
        command = f'"{self.ffmpeg_path}" -i "{self.input}" -ss {self.inpoint} -to {self.outpoint} -vf {self.mpdecimate_params} -an -y "{self.output}" -v quiet -stats'
        
        subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return os.path.normpath(self.output)
        
        