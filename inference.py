import cv2
import os
from threading import Thread
import torch
import time

def process_video(video_file, output_path, model, height, width):
    video_stream = VideoDecodeStream(video_file, width, height)
    video_stream.start()
    
    i = 0
    prev_frame = video_stream.read()
    time.sleep(1)
    while True:
        next_frame = video_stream.read()
        if next_frame is None:
            break

        prev_frame_tensor = torch.from_numpy(prev_frame).unsqueeze(0).float().cuda()
        next_frame_tensor = torch.from_numpy(next_frame).unsqueeze(0).float().cuda()
        
        with torch.no_grad():
            # Pass the previous and next frames to the model
            interpolated_frame_tensor = model.execute(prev_frame_tensor, next_frame_tensor, timestep=1.0)

        interpolated_frame = interpolated_frame_tensor
        
        cv2.imwrite(os.path.join(output_path, f'frame_{i}.png'), interpolated_frame)
        
        i += 1
        prev_frame = next_frame
    
class VideoDecodeStream:
    def __init__(self, video_file, width, height):
        self.vcap = cv2.VideoCapture(video_file)
        self.width = width
        self.height = height
        self.decode_buffer = []
        self.grabbed , self.frame = self.vcap.read()
        #self.frame = cv2.resize(self.frame, (self.width, self.height))
        self.decode_buffer.append(self.frame)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True 

    def start(self):
        self.stopped = False
        self.thread.start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.vcap.read()
            if not grabbed:
                print('The video buffering has been finished')
                self.stopped = True
                break
            #frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.decode_buffer.append(frame)
        self.vcap.release()

    def read(self):
        return self.decode_buffer.pop(0) if self.decode_buffer else None

    def stop(self):
        self.stopped = True 

    def get_fps(self):
        return self.vcap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return self.vcap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_fourcc(self):
        return cv2.VideoWriter_fourcc(*'mp4v')