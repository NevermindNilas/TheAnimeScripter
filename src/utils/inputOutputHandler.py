import os
from src.utils.generateOutput import outputNameGenerator

ALLOWED = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".gif"]


def genOutputHandler(video, output, outputPath, args):
    if video.endswith(".gif"):
        if output is None or os.path.isdir(output):
            return os.path.join(output or outputPath, outputNameGenerator(args, video))
    else:
        if output is None:
            return os.path.join(outputPath, outputNameGenerator(args, video))
        elif output.endswith(tuple(ALLOWED)):
            return output
        elif not output.endswith("\\"):
            tempOutput = output + "\\"
            if os.path.isdir(tempOutput):
                return os.path.join(tempOutput, outputNameGenerator(args, video))
        else:
            raise FileNotFoundError(f"File {output} does not exist")
    return output


def encoderChecker(video, encodeMethod, customEncoder):
    if (
        video.endswith(".webm")
        and customEncoder is None
        and encodeMethod not in ["vp9", "qsv_vp9", "av1"]
    ):
        return "vp9"
    return encodeMethod


def handleInputOutputs(args, isFrozen, outputPath):
    """
    Handles input and output paths for video processing.

    Args:
        outputPath (str): The path to the output directory
        isFrozen (bool): The frozen state of the application, Pyinstaller or Python
        args (argparse.Namespace): The arguments passed to the application

    Returns:
        dict: A dictionary containing videoPath, outputPath, encodeMethod, and customEncoder.
    """
    videos = os.path.abspath(args.input)
    output = args.output
    encodeMethod = args.encode_method
    customEncoder = args.custom_encoder

    os.makedirs(outputPath, exist_ok=True)
    if output and os.path.isdir(output):
        os.makedirs(output, exist_ok=True)

    result = {}
    index = 1

    if os.path.isdir(videos):
        video_files = [
            os.path.join(videos, f)
            for f in os.listdir(videos)
            if os.path.splitext(f)[1] in ALLOWED
        ]
    elif os.path.isfile(videos) and not videos.endswith(".txt"):
        video_files = [videos]
    else:
        if videos.endswith(".txt"):
            with open(videos, "r") as file:
                video_files = [line.strip().strip('"') for line in file.readlines()]
        else:
            video_files = videos.split(";")

    for video in video_files:
        if not os.path.exists(video):
            raise FileNotFoundError(f"File {video} does not exist")
        result[index] = {
            "videoPath": video,
            "outputPath": genOutputHandler(video, output, outputPath, args),
            "encodeMethod": encoderChecker(video, encodeMethod, customEncoder),
            "customEncoder": customEncoder,
        }
        index += 1

    return result
