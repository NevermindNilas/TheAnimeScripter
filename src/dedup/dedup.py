import subprocess
class DedupFFMPEG():
    def __init__(self, input, output, mpdecimate_params, ffmpeg_path):
        self.input = input
        self.output = output
        self.mpdecimate_params = mpdecimate_params
        self.ffmpeg_path = ffmpeg_path
        
    def run(self):
        ffmpeg_command = [self.ffmpeg_path, "-i", self.input, "-vf",
                          self.dedup_strenght, "-an", "-y", self.output]
        subprocess.Popen(
            ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

