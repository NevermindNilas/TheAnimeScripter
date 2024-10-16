import os
import sys

from src.utils.generateOutput import outputNameGenerator


def handleInputOutputs(args, isFrozen):
    """
    The premise goes as,

    For videos in videoss, return the videos as string to the path,

    this accounts for if the outputPath is not declared or if it's just a directory

    Args:
        input (str): The videos string
        output (str): The output name
        encodeMethod (str): The encoding method
        customEncoder (str): The custom encoder
        isFrozen (bool): The frozen state of the application, Pyinstaller or Python
        args (argparse.Namespace): The arguments passed to the application

    Returns:
        dict: A dictionary containing videoPath, outputPath, encodeMethod, and customEncoder.
    """

    videos = args.input
    output = args.output
    encodeMethod = args.encode_method
    customEncoder = args.custom_encoder

    outputPath = (
        os.path.dirname(sys.executable)
        if isFrozen
        else os.path.dirname(os.path.abspath(__file__))
    )
    AllowedExtensions = [
        ".mp4",
        ".mkv",
        ".webm",
        ".avi",
        ".mov",
        ".png",
        ".jpg",
        ".jpeg",
    ]
    result = {}
    videos = os.path.abspath(videos)

    # Redundant check, but it's here for safety
    os.makedirs(outputPath, exist_ok=True)

    if output is not None:
        output = os.path.abspath(output)
        os.makedirs(output, exist_ok=True)

    def genOutputHandler(video, output):
        if video.endswith((".jpg", ".jpeg", ".png")):
            if output is None:
                # Let FFMPEG handle the output names
                inputName = os.path.splitext(os.path.basename(video))[0]
                return os.path.join(
                    outputPath, f"{inputName}%04d{os.path.splitext(video)[1]}"
                )
            elif os.path.isdir(output):
                return os.path.join(output, outputNameGenerator(args, video))
        else:
            if output is None:
                return os.path.join(outputPath, outputNameGenerator(args, video))
            elif os.path.isdir(output):
                return os.path.join(output, outputNameGenerator(args, video))
            elif os.path.isfile(output):
                return output
        return output

    def encoderChecker(video, encodeMethod):
        if video.endswith((".webm")):
            if customEncoder is None and encodeMethod not in [
                "vp9",
                "qsv_vp9",
                "av1",
            ]:
                return "vp9"
            else:
                return encodeMethod
        return encodeMethod

    index = 1
    if os.path.isdir(videos):
        video_files = [
            os.path.join(videos, f)
            for f in os.listdir(videos)
            if os.path.splitext(f)[1] in AllowedExtensions
        ]
        for video in video_files:
            result[index] = {
                "videoPath": video,
                "outputPath": genOutputHandler(video, output),
                "encodeMethod": encoderChecker(video, encodeMethod),
                "customEncoder": customEncoder,
            }
            index += 1

    elif os.path.isfile(videos) and not videos.endswith((".txt")):
        result[index] = {
            "videoPath": videos,
            "outputPath": genOutputHandler(videos, output),
            "encodeMethod": encoderChecker(videos, encodeMethod),
            "customEncoder": customEncoder,
        }

    else:
        if videos.endswith(".txt"):
            with open(videos, "r") as file:
                videoFiles = [line.strip().strip('"') for line in file.readlines()]
        else:
            videoFiles = videos.split(";")

        for video in videoFiles:
            if not os.path.exists(video):
                raise FileNotFoundError(f"File {video} does not exist")
            result[index] = {
                "videoPath": video,
                "outputPath": genOutputHandler(video, output),
                "encodeMethod": encoderChecker(video, encodeMethod),
                "customEncoder": customEncoder,
            }
            index += 1

    # Returns a dictionary with indexed entries containing videoPath, outputPath, encodeMethod, and customEncoder
    return result
