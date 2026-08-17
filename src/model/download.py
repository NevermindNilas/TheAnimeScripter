"""Network and filesystem I/O for model downloads."""

import logging
import os

from src.infra.logAndPrint import logAndPrint

from .registry import (
    DEPTHV2URLSMALL,
    SUDOURL,
    TASURL,
    TRANSNETV2URL,
    modelsMap,
    weightsDir,
)


def downloadAndLog(
    model: str, filename: str, download_url: str, folderPath: str, retries: int = 3
):
    from src.constants import ADOBE

    if ADOBE:
        from src.server.aeComms import progressState

    import zipfile
    from http.client import IncompleteRead
    from urllib.error import HTTPError
    from urllib.request import urlopen

    # Imported lazily so registry-only consumers (modelsList/modelsMap, the
    # drift-guard tests) don't drag in barflow at module import.
    from src.infra.progressBarLogic import ProgressBarDownloadLogic

    tempFolder = os.path.join(folderPath, "TEMP")
    os.makedirs(tempFolder, exist_ok=True)
    # Process-unique staging name: two concurrent TAS runs fetching the same
    # missing model must not share a temp file, or they interleave writes and
    # the loser dies renaming a file the winner still holds open.
    tempFilePath = os.path.join(tempFolder, f"{filename}.{os.getpid()}.part")
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
                destPath = os.path.join(folderPath, filename)
                if os.path.exists(destPath):
                    # A concurrent TAS process committed the same model while
                    # we were downloading. Keep its copy; ours is identical.
                    os.remove(tempFilePath)
                else:
                    # os.replace is atomic and overwrites, so a peer committing
                    # between the check above and here cannot fail the commit.
                    os.replace(tempFilePath, destPath)

            try:
                os.rmdir(tempFolder)
            except OSError:
                pass

            toLog = f"Downloaded {model.capitalize()} model to: {os.path.join(folderPath, filename)}"
            logging.info(toLog)
            logAndPrint(toLog, colorFunc="green")

            return os.path.join(folderPath, filename)

        except (
            # OSError covers URLError/HTTPError/ConnectionError/TimeoutError
            # plus filesystem failures (Windows sharing violations, ENOSPC),
            # which used to escape the loop with no retry and no cleanup.
            OSError,
            zipfile.BadZipFile,
            IncompleteRead,
        ) as e:
            logging.error(f"Error during download: {e}")
            # Never remove the destination here: with the atomic commit above a
            # file at destPath is always complete, and possibly a concurrent
            # process's copy.
            try:
                if os.path.exists(tempFilePath):
                    os.remove(tempFilePath)
            except Exception:
                pass
            # A 404 means the asset was never published -- a CLI choice whose
            # weight is missing from the release, not a flaky network. Retrying
            # it twice more only delays an identical failure, and the raw
            # "HTTP Error 404" names neither the model nor the file.
            if isinstance(e, HTTPError) and e.code == 404:
                logAndPrint(
                    f"'{model}' cannot be downloaded: {filename} is not "
                    f"published on the model host ({download_url} returned "
                    "404). This CLI choice has no working weights; please "
                    "pick another method and report it.",
                    colorFunc="red",
                )
                raise
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


# The five checkpoints src/gmfss/model/GMFSS.py torch.loads. The bundle also
# ships *_base.pkl variants that nothing loads, so requiring them would force a
# pointless 110 MB re-download. Kept here rather than next to the loader because
# src/gmfss/ is a vendored tree that ruff and ty both exclude, so drift there
# gets no lint coverage.
GMFSS_REQUIRED_MEMBERS = (
    "rife.pkl",
    "flownet.pkl",
    "metric_union.pkl",
    "feat_union.pkl",
    "fusionnet_union.pkl",
)


def resolveWeightDir(
    subdir: str,
    requiredMembers: tuple[str, ...],
    downloadModel: str | None = None,
) -> str:
    """Return a multi-file weight folder, downloading it if it is incomplete.

    :func:`resolveWeightPath`'s sibling for models that ship as a zip of
    several files rather than one checkpoint, so there is no single filename to
    test for.

    Guarding on the folder's existence is not enough: ``downloadModels``
    creates ``weights/<model>/`` before it fetches a byte, and extracts the zip
    straight into it, so a Ctrl-C, a dropped connection or a full disk leaves a
    folder that exists and satisfies the guard but holds nothing the loader
    needs (often just ``TEMP/``). Every later run then took the "already have
    it" branch and never retried, which bricked the model permanently until the
    user found and deleted the folder by hand -- nothing told them to.
    Checking the members the loader actually opens makes that state self-heal,
    and covers a half-finished extraction as well as an empty folder.
    """
    modelDir = os.path.join(weightsDir, subdir)

    def missingMembers():
        return [
            m for m in requiredMembers if not os.path.exists(os.path.join(modelDir, m))
        ]

    if not missingMembers():
        return modelDir

    if os.path.isdir(modelDir):
        logging.info(
            f"{subdir} weights are incomplete (missing "
            f"{', '.join(missingMembers())}); re-downloading."
        )

    downloadModels(model=downloadModel if downloadModel is not None else subdir)

    # Return the directory we actually checked, not one derived from the
    # download's return value: downloadAndLog rewrites `filename` from the zip's
    # namelist, and returns None when the retry loop is exhausted -- deriving
    # the path from it could point somewhere else entirely, or raise TypeError.
    stillMissing = missingMembers()
    if stillMissing:
        # Without this the incomplete folder is handed back and the loader dies
        # on a missing file -- and because the guard would fail again next run,
        # every subsequent run re-downloads forever: the mirror image of the bug
        # this function exists to fix.
        raise FileNotFoundError(
            f"{subdir} weights are still incomplete after downloading (missing "
            f"{', '.join(stillMissing)} in {modelDir}). Delete that folder and "
            "retry."
        )
    return modelDir


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

    downloadModel = model.removesuffix("-mps") if model.endswith("-mps") else model
    registryModel = {
        "og_small_v2": "small_v2",
    }.get(downloadModel, downloadModel)

    filename = modelsMap(registryModel, upscaleFactor, modelType, half, ensemble)
    if registryModel.endswith("-tensorrt") or registryModel.endswith("-directml"):
        if "rife" in registryModel:
            folderName = registryModel.replace("-tensorrt", "")

        else:
            folderName = registryModel.replace("-tensorrt", "-onnx").replace(
                "-directml", "-onnx"
            )
    else:
        folderName = registryModel

    folderPath = os.path.join(weightsDir, folderName)
    os.makedirs(folderPath, exist_ok=True)

    if registryModel in [
        "shift_lpips-tensorrt",
        "shift_lpips-directml",
    ]:
        fullUrl = f"{SUDOURL}{filename}"
        try:
            # Just adds a redundant check if sudo decides to nuke his models.
            return downloadAndLog(downloadModel, filename, fullUrl, folderPath)
        except Exception as e:
            logging.warning(f"Failed to download from SUDOURL: {e}")
            fullUrl = f"{TASURL}{filename}"
            return downloadAndLog(downloadModel, filename, fullUrl, folderPath)

    elif registryModel == "transnetv2":
        fullUrl = f"{TRANSNETV2URL}{filename}"

    elif registryModel == "small_v2":
        fullUrl = f"{DEPTHV2URLSMALL}{filename}"

    else:
        fullUrl = f"{TASURL}{filename}"

    return downloadAndLog(downloadModel, filename, fullUrl, folderPath)
