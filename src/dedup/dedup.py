import subprocess
import os
class Dedup():
    def __init__(self, video, output, kind_model):
        self.video = video
        self.output = output
        self.kind_model = kind_model
        
        if self.kind_model == "ffmpeg":
            self.ffmpeg()
        elif self.kind_model == "ssim":
            self.ssim()
        elif self.kind_model == "hash":
            self.hash()
        elif self.kind_model == "vmaf":
            self.vmaf()
        else:
            print("No deduplication method was selected or it was invalid, defaulting to ffmpeg")
            self.ffmpeg()
    
    def ffmpeg(self):
        try:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            root_dir = os.path.dirname(script_dir)
            ffmpeg_path = os.path.join(root_dir, "ffmpeg", "ffmpeg.exe")
            ffmpeg_command = f"{ffmpeg_path} -i \"{self.video}\" -vf \"mpdecimate,setpts=N/FRAME_RATE/TB\" -an -y \"{self.output}\" -v quiet -stats"
            process = subprocess.run(ffmpeg_command)
                
        except:
            raise Exception("FFmpeg failed to deduplicate the video, please try again or use another deduplication method.")
        
    def hash(self):
        pass

    def vmaf(self):
        pass 

    def ssim(self):
        pass


    