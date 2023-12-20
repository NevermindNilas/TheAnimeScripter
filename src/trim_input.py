
class trim_input():
    def __init__(self, input, output, inpoint, outpoint, Do_not_process):
        import os
        self.input = input
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.output = output
        self.Do_not_process = Do_not_process
        
        abs_path = os.path.abspath(__file__)
        self.dir_path = os.path.dirname(abs_path)
        
        if not self.Do_not_process:
            filename_without_ext = os.path.splitext(os.path.basename(self.output))[0]
            dirname = os.path.dirname(self.output)
            self.output = os.path.join(dirname, filename_without_ext + "_trimmed.mp4")
            
        self.ffmpeg_path = os.path.join(self.dir_path, "ffmpeg", "ffmpeg.exe")
        
    def run(self):
        import subprocess

        command = f'"{self.ffmpeg_path}" -i "{self.input}" -ss {self.inpoint} -to {self.outpoint} "{self.output}" -v quiet -stats -y'
        
        subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return self.output

class trim_input_dedup():
    def __init__(self, input, output, inpoint, outpoint, Do_not_process):
        import os
        self.input = input
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.output = output
        self.Do_not_process = Do_not_process
        
        abs_path = os.path.abspath(__file__)
        self.dir_path = os.path.dirname(abs_path)
        
        if not self.Do_not_process:
            filename_without_ext = os.path.splitext(os.path.basename(self.output))[0]
            dirname = os.path.dirname(self.output)
            self.output = os.path.join(dirname, filename_without_ext + "_trimmed_dedup.mp4")
            
        self.ffmpeg_path = os.path.join(self.dir_path, "ffmpeg", "ffmpeg.exe")
            
    def run(self):
        import subprocess
        
        
        command = f'"{self.ffmpeg_path}" -i "{self.input}" -ss {self.inpoint} -to {self.outpoint} -vf "mpdecimate=hi=64*24:lo=64*12:frac=0.1,setpts=N/FRAME_RATE/TB" -an -y "{self.output}" -v quiet -stats'
        
        subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return self.output
        
        