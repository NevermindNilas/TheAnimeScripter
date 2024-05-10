import requests
import logging
import json
import pathlib
from tqdm import tqdm
import subprocess
import os
import shutil

from .coloredPrints import green

URL = r"https://api.github.com/repos/NevermindNilas/TheAnimeScripter/releases/latest"


def updateScript(scriptVersion, mainPath):
    """
    Update the script if the version is outdated
    """
    try:
        print(green("Checking for updates..."))
        response = requests.get(URL)
        data = json.loads(response.text)
        latestVersion = data["tag_name"]

        scriptVersion = "v" + scriptVersion
        if latestVersion != scriptVersion:
            answer = input(
                f"New version {latestVersion} has been found, current version: {scriptVersion}. Do you want to update? (y/n): "
            )

            if answer.lower() != "y":
                logging.info("User chose not to update")
                return

            logging.info(
                f"New version found: {latestVersion}, downloading and installing"
            )

            url7zr = "https://www.7-zip.org/a/7zr.exe"
            response7zr = requests.get(url7zr)

            pathTo7zr = os.path.join(mainPath, "7zr.exe")

            with open(pathTo7zr, "wb") as file:
                file.write(response7zr.content)

            for asset in data["assets"]:
                if asset["content_type"] == "application/x-7z-compressed":
                    downloadUrl = asset["browser_download_url"]
                    response = requests.get(downloadUrl, stream=True)
                    if response.status_code == 200:
                        fileName = downloadUrl.split("/")[-1]
                        fileNameWithoutExt = os.path.splitext(fileName)[0]
                        parentDir = pathlib.Path(mainPath).parent
                        filePath = parentDir / fileName
                        totalSizeInBytes = int(
                            response.headers.get("content-length", 0)
                        )
                        blockSize = 1024
                        progressBar = tqdm(
                            total=totalSizeInBytes,
                            unit="iB",
                            unit_scale=True,
                            bar_format="{l_bar}%s{bar}%s{r_bar}"
                            % ("\033[32m", "\033[0m"),
                        )
                        with open(filePath, "wb") as file:
                            for data in response.iter_content(blockSize):
                                progressBar.update(len(data))
                                file.write(data)
                        progressBar.close()
                        if totalSizeInBytes != 0 and progressBar.n != totalSizeInBytes:
                            print("ERROR, something went wrong")

                        logging.info(
                            "Download complete, extracting files, depending on the size of the update this might take a while"
                        )
                        print(
                            green(
                                "Download complete, extracting files, depending on the size of the update this might take a while"
                            )
                        )

                        extractDir = parentDir / fileNameWithoutExt
                        extractDir.mkdir(exist_ok=True)

                        subprocess.run(
                            [
                                str(pathTo7zr),
                                "x",
                                str(filePath),
                                "-o" + str(extractDir),
                                "-aoa",
                            ]
                        )

                        filePath.unlink()
                        break
            
            """
            TO:DO - WIP, need to fix this
            answer = input("Do you wish to copy over the models and ffmpeg files to the new directory? (y/n): ")
            if answer.lower() == "y":
                weightsDir = os.path.join(parentDir, "weights")
                ffmpegDir = os.path.join(parentDir, "ffmpeg")
                shutil.copytree(weightsDir, os.path.join(extractDir, "weights"))
                shutil.copytree(ffmpegDir, os.path.join(extractDir, "ffmpeg"))
            """

            print(
                green(
                    f"Update downloaded to {extractDir}, if you are an After Effects user, please update the After Effects UI script as well. You may copp over the models found in the src/weights folder and ffmpeg to the new directory if you wish to do so."
                )
            )

        else:
            print(green("No updates found, script is up to date"))
    except Exception as e:
        logging.error(f"An error occurred: {e}")
