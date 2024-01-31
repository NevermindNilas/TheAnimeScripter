import cv2
import logging

def getVideoMetadata(input, inPoint, outPoint):
    cap = cv2.VideoCapture(input)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    duration = nframes / fps if fps else 0

    in_out_duration = (outPoint - inPoint) / fps if fps else 0

    logging.info(f"Width: {width}")
    logging.info(f"Height: {height}")
    logging.info(f"AspectRatio: {width/height}")
    logging.info(f"FPS: {fps}")
    logging.info(f"Number of frames: {nframes}")
    logging.info(f"Codec: {codec}")
    logging.info(f"Duration: {duration} seconds")
    logging.info(f"In-Out Duration: {in_out_duration} seconds")

    cap.release()

    return width, height, fps, nframes