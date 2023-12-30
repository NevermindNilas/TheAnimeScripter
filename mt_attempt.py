import os
import logging
import subprocess
import argparse
import numpy as np

from moviepy.editor import VideoFileClip
from tqdm import tqdm
from multiprocessing import Process, Queue

def main(init_args, ffmpeg_path, width, height, fps, nframes):
    
    read_buffer = Queue(maxsize=500)
    write_buffer = Queue(maxsize=500)
    
    procs = []
    interpolate_process,  upscale_process, new_width, new_height, fps = intitialize_models(fps, width, height, init_args)
    
    proc = Process(target=build_read_buffer, args=(
        read_buffer, args.input, ffmpeg_path, width, height))
    procs.append(proc)
    proc.start()

    # I will want to eventually multi-thread this, but for now it's fine
    proc = Process(target=start_process, args=(read_buffer, write_buffer, upscale_process, interpolate_process, init_args.interpolate_factor, init_args.upscale, init_args.interpolate))
    procs.append(proc)
    proc.start()

    proc = Process(target=clear_write_buffer, args=(write_buffer, new_width, new_height, args.output, fps, ffmpeg_path, nframes))
    procs.append(proc)
    proc.start()

    for proc in procs:
        proc.join()

def get_ffmpeg_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ffmpeg_path = os.path.join(dir_path, "ffmpeg", "ffmpeg.exe")
    
    if not os.path.exists(ffmpeg_path):
        print("Couldn't find FFMPEG, downloading it now")
        print("This might add an aditional 1-5 seconds to the startup time of the process until FFMPEG is downloaded and caches are built, but it will only happen once")
        logging.info("The user doesn't have FFMPEG, downloading it now")
        ffmpeg_bat_location = os.path.join(dir_path, "get_ffmpeg.bat")
        subprocess.call(ffmpeg_bat_location, shell=True)
        
    return ffmpeg_path

def get_video_metadata(input):
    clip = VideoFileClip(input)
    width = clip.size[0]
    height = clip.size[1]
    fps = clip.fps
    nframes = clip.reader.nframes

    logging.info(
        f"Video Metadata: {width}x{height} @ {fps}fps, {nframes} frames")

    return width, height, fps, nframes

def intitialize_models(fps, width, height, args):

    fps = fps * args.interpolate_factor if args.interpolate == True else fps
    if args.upscale:
        logging.info(
            f"Upscaling to {width*args.upscale_factor}x{height*args.upscale_factor}")
        
        match args.upscale_method:
            case "shufflecugan" | "cugan":
                from src.cugan.cugan import Cugan
                
                upscale_process = Cugan(
                    args.upscale_method, int(args.upscale_factor), args.cugan_kind, args.half, width, height)
                
            case "cugan-amd":
                from src.cugan.cugan import CuganAMD
                upscale_process = CuganAMD(
                    args.nt, args.upscale_factor
                )
                
            case "compact" | "ultracompact" | "superultracompact":
                from src.compact.compact import Compact
                upscale_process = Compact(
                    args.upscale_method, args.half)
                
            case "swinir":
                from src.swinir.swinir import Swinir
                upscale_process = Swinir(
                    args.upscale_factor, args.half, width, height)
                
            case _:
                logging.info(
                    f"There was an error in choosing the upscale method, {args.upscale_method} is not a valid option")
    
        width *= args.upscale_factor
        height *= args.upscale_factor
    else:
        upscale_process = None
           
    if args.interpolate:
        from src.rife.rife import Rife
        
        UHD = True if width >= 3840 and height >= 2160 else False
        interpolate_process = Rife(
            int(args.interpolate_factor), args.half, width, height, UHD)
    else:
        interpolate_process = None
    
    return interpolate_process, upscale_process, width, height, fps
       
def build_read_buffer(read_buffer, input, ffmpeg_path, width, height):
    ffmpeg_command = [
        ffmpeg_path,
        "-i", str(input),
    ]

    ffmpeg_command.extend([
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "quiet",
        "-stats",
        "-",
    ])

    process = subprocess.Popen(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    logging.info(f"Running command: {ffmpeg_command}")

    frame_size = width * height * 3
    frame_count = 0

    for chunk in iter(lambda: process.stdout.read(frame_size), b''):
        if len(chunk) != frame_size:
            logging.error(
                f"Read {len(chunk)} bytes but expected {frame_size}")
            break
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
            (height, width, 3))
        read_buffer.put(frame)
        frame_count += 1
        
    stderr = process.stderr.read().decode()
    if stderr:
        if "bitrate=" not in stderr:
            logging.error(f"ffmpeg error: {stderr}")
            
    process.stdout.close()
    process.stderr.close()
    process.terminate()

    for _ in range(1):
        read_buffer.put(None)

    logging.info(f"Read {frame_count} frames")


def start_process(read_buffer, write_buffer, upscale_process, interpolate_process, interpolate_factor, upscale, interpolate):
    prev_frame = None
    try:
        while True:
            frame = read_buffer.get()
            if frame is None:
                read_buffer.put(None)
                break
            
            if upscale == True:
                frame = upscale_process.run(frame)
                
            if interpolate == True:
                if prev_frame is not None:
                    results = interpolate_process.run(prev_frame, frame, interpolate_factor)
                    for result in results:
                        write_buffer.put(result)
                    prev_frame = frame
                else:
                    prev_frame = frame
            
            write_buffer.put(frame)
            
    except Exception as e:
        logging.exception("An error occurred during processing")

def clear_write_buffer(write_buffer, width, height, output, fps, ffmpeg_path, nframes):
    pbar = tqdm(total=nframes, desc="Writing frames")
    command = [ffmpeg_path,
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-s', f'{width}x{height}',
           '-pix_fmt', 'rgb24',
           '-r', str(fps),
           '-i', '-',
           '-an',
           '-c:v', 'libx264',
           '-preset', 'veryfast',
           '-crf', '15',
           '-tune', 'animation',
           '-movflags', '+faststart',
           output]
    
    pipe = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            frame = write_buffer.get()
            if frame is None:
                break
            
            pipe.stdin.write(frame.tobytes())
            pbar.update(1)

    except Exception as e:
        logging.exception("An error occurred during writing")
    pipe.stdin.close()
    pipe.wait()


if __name__ == "__main__":

    log_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'log.txt')
   
    logging.basicConfig(filename=log_file_path, filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--interpolate", type=int, default=0)
    argparser.add_argument("--interpolate_factor", type=int, default=2)
    argparser.add_argument("--upscale", type=int, default=0)
    argparser.add_argument("--upscale_factor", type=int, default=2)
    argparser.add_argument("--upscale_method",  type=str,
                           default="ShuffleCugan")
    argparser.add_argument("--cugan_kind", type=str, default="no-denoise")
    argparser.add_argument("--dedup", type=int, default=0)
    argparser.add_argument("--dedup_method", type=str, default="ffmpeg")
    argparser.add_argument("--dedup_strenght", type=str, default="light")
    argparser.add_argument("--nt", type=int, default=1)
    argparser.add_argument("--half", type=int, default=1)
    argparser.add_argument("--inpoint", type=float, default=0)
    argparser.add_argument("--outpoint", type=float, default=0)
    argparser.add_argument("--sharpen", type=int, default=0)
    argparser.add_argument("--sharpen_sens", type=float, default=50)
    argparser.add_argument("--segment", type=int, default=0)
    argparser.add_argument("--scenechange", type=int, default=0)
    argparser.add_argument("--scenechange_sens", type=float, default=40)
    
    try:
        args = argparser.parse_args()
    except Exception as e:
        logging.info(e)

    # Whilst this is ugly, it was easier to work with the Extendscript interface this way
    args.interpolate = True if args.interpolate == 1 else False
    args.scenechange = True if args.scenechange == 1 else False
    args.sharpen = True if args.sharpen == 1 else False
    args.upscale = True if args.upscale == 1 else False
    args.segment = True if args.segment == 1 else False
    args.dedup = True if args.dedup == 1 else False
    args.half = True if args.half == 1 else False

    args.upscale_method = args.upscale_method.lower()
    args.dedup_strenght = args.dedup_strenght.lower()
    args.dedup_method = args.dedup_method.lower()
    args.cugan_kind = args.cugan_kind.lower()
    
    args.sharpen_sens /= 100  # CAS works from 0.0 to 1.0
    args.scenechange_sens /= 100 # same for scene change

    args.input = os.path.normpath(args.input)
    args.output = os.path.normpath(args.output)
    
    args_dict = vars(args)
    for arg in args_dict:
        logging.info(f"{arg}: {args_dict[arg]}")

    if args.output and not os.path.isabs(args.output):
        dir_path = os.path.dirname(args.input)
        args.output = os.path.join(dir_path, args.output)

    if args.upscale_method in ["shufflecugan", "compact", "ultracompact", "superultracompact", "swinir"] and args.upscale_factor != 2:
        logging.info(
            f"{args.upscale_method} only supports 2x upscaling, setting upscale_factor to 2, please use Cugan for 3x/4x upscaling")
        args.upscale_factor = 2

    if args.upscale_factor not in [2, 3, 4]:
        logging.info(
            f"{args.upscale_factor} is not a valid upscale factor, setting upscale_factor to 2")
        args.upscale_factor = 2

    if args.nt > 1:
        logging.info(
            "Multi-threading is not supported yet, setting nt back to 1")
        args.nt = 1

    dedup_strenght_list = {
        "light": "mpdecimate=hi=64*24:lo=64*12:frac=0.1,setpts=N/FRAME_RATE/TB",
        "medium": "mpdecimate=hi=64*100:lo=64*35:frac=0.2,setpts=N/FRAME_RATE/TB",
        "high": "mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB"
    }

    args.dedup_strenght = dedup_strenght_list[args.dedup_strenght]

    ffmpeg_path = get_ffmpeg_path()
    width, height, fps, nframes = get_video_metadata(args.input)
    
    main(args, ffmpeg_path, width, height, fps, nframes)
