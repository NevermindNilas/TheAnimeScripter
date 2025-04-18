import inquirer
import subprocess
import os

from pathlib import Path
from anipy_api.provider.providers import GoGoProvider
from anipy_api.anime import Anime
from anipy_api.provider import LanguageTypeEnum
from anipy_api.download import Downloader
from src.utils.coloredPrints import yellow
from src.constants import FFMPEGPATH


def initializeProvider():
    return GoGoProvider()


def getAnimeName():
    return input("Enter the anime name you wish to download: ")


def searchAnime(provider, animeName):
    return provider.get_search(animeName)


def convertToAnimeObjects(provider, results):
    animeList = []
    for r in results:
        animeList.append(Anime.from_search_result(provider, r))
    return animeList


def selectAnime(animeList):
    questions = [
        inquirer.List(
            "anime",
            message=f"Select the anime [1-{len(animeList)}]",
            choices=[f"{idx + 1}: {anime}" for idx, anime in enumerate(animeList)],
        ),
    ]
    answers = inquirer.prompt(questions)
    selection = int(answers["anime"].split(":")[0]) - 1
    return selection


def getEpisodes(selectedAnime):
    return selectedAnime.get_episodes(lang=LanguageTypeEnum.SUB)


def selectEpisode(episodes):
    episodeQuestions = [
        inquirer.List(
            "episode",
            message=f"Select the episode [1-{len(episodes)}]",
            choices=[f"Episode {ep}" for ep in episodes],
        ),
    ]
    episodeAnswers = inquirer.prompt(episodeQuestions)
    episodeSelection = int(episodeAnswers["episode"].split(" ")[1])
    return episodeSelection


def getVideoStream(selectedAnime, episodeSelection):
    return selectedAnime.get_video(
        episode=episodeSelection,
        lang=LanguageTypeEnum.DUB,
        preferred_quality=1080,
    )


def progressCallback(percentage: float):
    print(f"Progress: {percentage:.1f}%", end="\r")


def infoCallback(message: str):
    print(f"Message from the downloader: {message}")


def errorCallback(message: str):
    print(f"Soft error from the downloader: {message}")


def downloadEpisode(episodeStream, outputPath: str = "~/Downloads"):
    downloader = Downloader(progressCallback, infoCallback, errorCallback)
    downloadPath = downloader.download(
        stream=episodeStream,
        download_path=Path(outputPath),
        # container=".mkv",
        max_retry=3,
        ffmpeg=False,
    )
    print(f"Downloaded to: {downloadPath}")
    return downloadPath


def aniPyHandler(
    outputPath: str = "~/Downloads",
):
    print(
        yellow(
            "This feature is in Beta, any and all bug reports will be ignored. Use it at your own risk, please read the license for more info."
        )
    )
    provider = initializeProvider()
    downloadSuccessful = False

    while not downloadSuccessful:
        animeName = getAnimeName()
        results = searchAnime(provider, animeName)

        if not results:
            print(yellow("No results found. Please try again."))
            continue

        animeList = convertToAnimeObjects(provider, results)
        selection = selectAnime(animeList)

        if 0 <= selection < len(animeList):
            selectedAnime = animeList[selection]
            episodes = getEpisodes(selectedAnime)
            episodeSelection = selectEpisode(episodes)
            episodeStream = getVideoStream(selectedAnime, episodeSelection)
            downloadPath = downloadEpisode(episodeStream, outputPath)
            if downloadPath:
                downloadSuccessful = True
        else:
            print("Invalid selection.")

    downloadPath = str(downloadPath)
    if not downloadPath.endswith(".mp4"):
        downloadPath = handleConversion(downloadPath, outputPath)

    return downloadPath


def handleConversion(downloadPath: str, outputPath: str = "~/Downloads"):
    print(
        yellow(
            f"Due to limitations in TAS' backend, we are converting {downloadPath} to mp4..."
        )
    )

    if not outputPath.endswith(".mp4"):
        desiredPath = os.path.join(outputPath, "converted.mp4")
    else:
        desiredPath = outputPath

    command = [
        FFMPEGPATH,
        "-i",
        downloadPath,
        "-c",
        "copy",
        "-loglevel",
        "error",
        desiredPath,
        "-y",
    ]
    subprocess.run(command)
    print(f"Converted to: {desiredPath}")

    try:
        os.remove(downloadPath)
    except Exception as e:
        print(f"Failed to remove {downloadPath}: {e}")

    return desiredPath
