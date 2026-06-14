"""Network and filesystem I/O for model downloads.

This module contains:
- downloadAndLog() - handles actual file downloads with progress bars
- downloadModels() - orchestrates model downloads
- resolveWeightPath() - resolves local path or triggers download
"""

import os
import logging
import zipfile

from src.infra.logAndPrint import logAndPrint

from .registry import (
    weightsDir,
    TASURL,
    SUDOURL,
    TRANSNETV2URL,
    DEPTHV2URLSMALL,
    DEPTHV2URLBASE,
    DEPTHV2URLLARGE,
    DEPTHV2URLGIANT,
    modelsMap,
)


def downloadAndLog(
    model: str, filename: str, download_url: str, folderPath: str, retries: int = 3
):
    from src.constants import ADOBE

    if ADOBE:
        from src.server.aeComms import progressState

    from urllib.request import urlopen
    from urllib.error import URLError, HTTPError
    from http.client import IncompleteRead

    # Imported lazily so registry-only consumers (modelsList/modelsMap, the
    # drift-guard tests) don't drag in barflow at module import.
    from src.infra.progressBarLogic import ProgressBarDownloadLogic

    tempFolder = os.path.join(folderPath, "TEMP")
    os.makedirs(tempFolder, exist_ok=True)
    tempFilePath = os.path.join(tempFolder, filename)
    downloadedBytes = 0
    totalSizeInBytes = 0
    if ADOBE:
        progressState.update(
            {
                "status": f"Downloading model {os.path.basename(filename)}.",
            }
        )

    for attempt in range(retries):
        try:
            if os.path.exists(os.path.join(folderPath, filename)):
                toLog = f"{model.upper()} model already exists at: {os.path.join(folderPath, filename)}"
                logging.info(toLog)
                return os.path.join(folderPath, filename)

            toLog = f"Downloading {model.upper()} model... (Attempt {attempt + 1}/{retries})"
            logging.info(toLog)

            try:
                if os.path.exists(tempFilePath):
                    os.remove(tempFilePath)
            except Exception:
                pass

            response = urlopen(download_url)

            if response.getcode() != 200:
                raise HTTPError(download_url, response.getcode(), None, None, None)

            try:
                totalSizeInBytes = int(response.headers.get("content-length", 0))
                totalSizeInMb = totalSizeInBytes / (1024 * 1024)  # Convert bytes to MB
            except Exception as e:
                totalSizeInBytes = 0  # If there's an error, default to 0 MB
                totalSizeInMb = 0
                logging.error(e)

            loggedPercentages = set()
            downloadedBytes = (
                0  # reset per attempt so the size check below is correct on retries
            )

            try:
                with ProgressBarDownloadLogic(
                    totalSizeInBytes or 1,
                    title=f"Downloading {model.upper()} model... (Attempt {attempt + 1}/{retries})",
                ) as bar:
                    with open(tempFilePath, "wb") as file:
                        while True:
                            data = response.read(1024 * 1024)
                            if not data:
                                break
                            file.write(data)
                            downloadedBytes += len(data)
                            bar(len(data))

                            if totalSizeInBytes > 0:
                                currentMb = downloadedBytes / (1024 * 1024)
                                currentPercentage = int(
                                    (downloadedBytes / totalSizeInBytes) * 100
                                )

                                for milestone in [20, 40, 60, 80, 100]:
                                    if (
                                        currentPercentage >= milestone
                                        and milestone not in loggedPercentages
                                    ):
                                        logging.info(
                                            f"Downloaded {milestone}% of {model.upper()} - {currentMb:.2f}/{totalSizeInMb:.2f} MB"
                                        )
                                        loggedPercentages.add(milestone)
            except UnicodeEncodeError as e:
                logging.warning(
                    f"Progress UI encoding issue on this console ({e}). Continuing without rich UI."
                )

            if totalSizeInBytes > 0 and downloadedBytes != totalSizeInBytes:
                # Server advertised a content-length we did not fully receive, so
                # the temp file is truncated. Trigger the retry/cleanup path
                # instead of committing a corrupt file to the weights cache.
                # NOTE: use ConnectionError, not IncompleteRead(int, int) -- the
                # latter's __repr__/__str__ does len(self.partial), which raises
                # TypeError when partial is an int, and that TypeError (not in the
                # except tuple below) would escape and abort the retry loop.
                raise ConnectionError(
                    f"Incomplete download: received {downloadedBytes} of "
                    f"{totalSizeInBytes} bytes"
                )

            if filename.endswith(".zip"):
                with zipfile.ZipFile(tempFilePath, "r") as zipRef:
                    zipRef.extractall(folderPath)

                    extractedFiles = zipRef.namelist()
                    onnxFiles = [f for f in extractedFiles if f.endswith(".onnx")]
                    if onnxFiles:
                        filename = onnxFiles[0]

                    elif any(f.endswith(".pth") for f in extractedFiles):
                        filename = [f for f in extractedFiles if f.endswith(".pth")][0]

                    elif os.path.exists(os.path.join(folderPath, filename[:-4])):
                        filename = filename[:-4]

                os.remove(tempFilePath)
            else:
                os.rename(tempFilePath, os.path.join(folderPath, filename))

            try:
                os.rmdir(tempFolder)
            except OSError:
                pass

            toLog = f"Downloaded {model.capitalize()} model to: {os.path.join(folderPath, filename)}"
            logging.info(toLog)
            logAndPrint(toLog, colorFunc="green")

            return os.path.join(folderPath, filename)

        except (
            URLError,
            HTTPError,
            zipfile.BadZipFile,
            IncompleteRead,
            ConnectionError,
            TimeoutError,
        ) as e:
            logging.error(f"Error during download: {e}")
            try:
                dest_path = os.path.join(folderPath, filename)
                if os.path.exists(dest_path):
                    os.remove(dest_path)
            except Exception:
                pass
            try:
                if os.path.exists(tempFilePath):
                    os.remove(tempFilePath)
            except Exception:
                pass
            if attempt == retries - 1:
                raise

    return None


def resolveWeightPath(
    subdir: str,
    filename: str,
    downloadModel: str = None,
    modelType: str = "pth",
    half: bool = True,
    ensemble: bool = False,
    upscaleFactor: int = 2,
) -> str:
    """
    Return the local weight file if present, otherwise download it.

    Args:
        subdir: Folder under `weightsDir` where the file is expected to live.
        filename: Expected filename inside `subdir`.
        downloadModel: Model identifier passed to `downloadModels` when missing.
            Defaults to `subdir` when not provided.
    """
    cachedPath = os.path.join(weightsDir, subdir, filename)
    if os.path.exists(cachedPath):
        return cachedPath
    return downloadModels(
        model=downloadModel if downloadModel is not None else subdir,
        upscaleFactor=upscaleFactor,
        modelType=modelType,
        half=half,
        ensemble=ensemble,
    )


def downloadModels(
    model: str = None,
    upscaleFactor: int = 2,
    modelType: str = "pth",
    half: bool = True,
    ensemble: bool = False,
) -> str:
    """
    Downloads the model.
    """
    os.makedirs(weightsDir, exist_ok=True)

    filename = modelsMap(model, upscaleFactor, modelType, half, ensemble)
    if model.endswith("-tensorrt") or model.endswith("-directml"):
        if "rife" in model:
            folderName = model.replace("-tensorrt", "")

        else:
            folderName = model.replace("-tensorrt", "-onnx").replace(
                "-directml", "-onnx"
            )
    else:
        folderName = model

    folderPath = os.path.join(weightsDir, folderName)
    os.makedirs(folderPath, exist_ok=True)

    if model in [
        "shift_lpips-tensorrt",
        "shift_lpips-directml",
    ]:
        fullUrl = f"{SUDOURL}{filename}"
        try:
            # Just adds a redundant check if sudo decides to nuke his models.
            return downloadAndLog(model, filename, fullUrl, folderPath)
        except Exception as e:
            logging.warning(f"Failed to download from SUDOURL: {e}")
            fullUrl = f"{TASURL}{filename}"
            return downloadAndLog(model, filename, fullUrl, folderPath)

    elif model == "transnetv2":
        fullUrl = f"{TRANSNETV2URL}{filename}"

    elif model == "small_v2":
        fullUrl = f"{DEPTHV2URLSMALL}{filename}"
    elif model == "base_v2":
        fullUrl = f"{DEPTHV2URLBASE}{filename}"
    elif model == "large_v2":
        fullUrl = f"{DEPTHV2URLLARGE}{filename}"
    elif model == "giant_v2":
        fullUrl = f"{DEPTHV2URLGIANT}{filename}"

    else:
        fullUrl = f"{TASURL}{filename}"

    return downloadAndLog(model, filename, fullUrl, folderPath)
