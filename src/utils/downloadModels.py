import os
import logging
from .coloredPrints import green
from src.utils.progressBarLogic import ProgressBarDownloadLogic

import platform

if platform.system() == "Windows":
    appdata = os.getenv("APPDATA")
    mainPath = os.path.join(appdata, "TheAnimeScripter")

    if not os.path.exists(mainPath):
        os.makedirs(mainPath)

    weightsDir = os.path.join(mainPath, "weights")
else:
    xdg_config_home = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    mainPath = os.path.join(xdg_config_home, "TheAnimeScripter")

    if not os.path.exists(mainPath):
        os.makedirs(mainPath)

    weightsDir = os.path.join(mainPath, "weights")

TASURL = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/"
DEPTHURL = (
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/"
)
SUDOURL = (
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"
)

DEPTHV2URLSMALL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
)
DEPTHV2URLBASE = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/"
)
DEPTHV2URLLARGE = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/"
)


def modelsList() -> list[str]:
    return [
        "shufflespan",
        "shufflespan-directml",
        "shufflespan-tensorrt",
        "aniscale2",
        "aniscale2-directml",
        "aniscale2-tensorrt",
        "open-proteus",
        "compact",
        "ultracompact",
        "superultracompact",
        "span",
        "shufflecugan",
        "segment",
        "segment-tensorrt",
        "segment-directml",
        "scunet",
        "dpir",
        "real-plksr",
        "nafnet",
        "anime1080fixer",
        "anime1080fixer-tensorrt",
        "anime1080fixer-directml",
        "rife",
        "rife4.6",
        "rife4.15-lite",
        "rife4.16-lite",
        "rife4.17",
        "rife4.18",
        "rife4.20",
        "rife4.21",
        "rife4.22",
        "rife4.22-lite",
        "rife4.25",
        "rife4.25-lite",
        "rife4.25-heavy",
        "rife_elexor",
        "rife4.6-tensorrt",
        "rife4.15-lite-tensorrt",
        "rife4.17-tensorrt",
        "rife4.18-tensorrt",
        "rife4.20-tensorrt",
        "rife4.21-tensorrt",
        "rife4.22-tensorrt",
        "rife4.22-lite-tensorrt",
        "rife4.25-tensorrt",
        "rife4.25-lite-tensorrt",
        "rife4.25-heavy-tensorrt",
        "rife_elexor-tensorrt",
        "rife-v4.6-ncnn",
        "rife-v4.15-lite-ncnn",
        "rife-v4.16-lite-ncnn",
        "rife-v4.17-ncnn",
        "rife-v4.18-ncnn",
        "rife-v4.20-ncnn",
        "rife-v4.21-ncnn",
        "rife-v4.22-ncnn",
        "rife-v4.22-lite-ncnn",
        "shufflecugan-directml",
        "open-proteus-directml",
        "compact-directml",
        "ultracompact-directml",
        "superultracompact-directml",
        "span-directml",
        "open-proteus-tensorrt",
        "shufflecugan-tensorrt",
        "compact-tensorrt",
        "ultracompact-tensorrt",
        "superultracompact-tensorrt",
        "span-tensorrt",
        "span-ncnn",
        "shufflecugan-ncnn",
        "small_v2",
        "base_v2",
        "large_v2",
        "small_v2-directml",
        "base_v2-directml",
        "large_v2-directml",
        "small_v2-tensorrt",
        "base_v2-tensorrt",
        "large_v2-tensorrt",
        "maxxvit-tensorrt",
        "maxxvit-directml",
        "shift_lpips-tensorrt",
        "shift_lpips-directml",
        "differential-tensorrt",
        "gmfss",
    ]


def modelsMap(
    model: str = "compact",
    upscaleFactor: int = 2,
    modelType: str = "pth",
    half: bool = True,
    ensemble: bool = False,
) -> str:
    """
    Maps the model to the corresponding filename.

    Args:
        model: The model to map.
        upscaleFactor: The upscale factor.
        modelType: The model type.
        half: Whether to use half precision or not.
        ensemble: Whether to use ensemble mode or not.
    """
    modelDict = {
        "shufflespan": {
            "pth": "sudo_shuffle_span_10.5m.pth",
            "onnx_fp16": "sudo_shuffle_span_op20_10.5m_1080p_fp16_op21_slim.onnx",
            "onnx_fp32": "sudo_shuffle_span_op20_10.5m_1080p_fp32_op21_slim.onnx",
        },
        "aniscale2": {
            "pth": "2x_AniScale2S_Compact_i8_60K.pth",
            "onnx_fp16": "2x_AniScale2S_Compact_i8_60K-fp16.onnx",
            "onnx_fp32": "2x_AniScale2S_Compact_i8_60K-fp32.onnx",
        },
        "open-proteus": {
            "pth": "2x_OpenProteus_Compact_i2_70K.pth",
            "onnx_fp16": "2x_OpenProteus_Compact_i2_70K-fp16.onnx",
            "onnx_fp32": "2x_OpenProteus_Compact_i2_70K-fp32.onnx",
        },
        "compact": {
            "pth": "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth",
            "onnx_fp16": "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k_clamp_fp16_op18_onnxslim.onnx",
            "onnx_fp32": "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k_clamp_op18_onnxslim.onnx",
        },
        "ultracompact": {
            "pth": "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k.pth",
            "onnx_fp16": "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k_clamp_fp16_op18_onnxslim.onnx",
            "onnx_fp32": "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k_clamp_op18_onnxslim.onnx",
        },
        "superultracompact": {
            "pth": "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k.pth",
            "onnx_fp16": "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k_clamp_fp16_op18_onnxslim.1.onnx",
            "onnx_fp32": "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k_clamp_op18_onnxslim.onnx",
        },
        "span": {
            "pth": "2x_ModernSpanimationV2.pth",
            "onnx_fp16": "2x_ModernSpanimationV2_fp16_op21_slim.onnx",
            "onnx_fp32": "2x_ModernSpanimationV2_fp32_op21_slim.onnx",
            "ncnn": "2x_modernspanimationv1.5-ncnn.zip",
        },
        "shufflecugan": {
            "pth": "sudo_shuffle_cugan_9.584.969.pth",
            "onnx_fp16": "sudo_shuffle_cugan_fp16_op18_clamped.onnx",
            "onnx_fp32": "sudo_shuffle_cugan_op18_clamped.onnx",
            "ncnn": "2xsudo_shuffle_cugan-ncnn.zip",
        },
        "segment": {
            "pth": "isnetis.ckpt",
        },
        "scunet": {
            "pth": "scunet_color_real_psnr.pth",
        },
        "dpir": {
            "pth": "drunet_deblocking_color.pth",
        },
        "real-plksr": {
            "pth": "1xDeJPG_realplksr_otf.pth",
        },
        "nafnet": {
            "pth": "NAFNet-GoPro-width64.pth",
        },
        "anime1080fixer": {
            "pth": "1x_Anime1080Fixer_SuperUltraCompact.pth",
            "onnx_fp16": "1x_Anime1080Fixer_SuperUltraCompact_op20_fp16_clamp.onnx",
            "onnx_fp32": "1x_Anime1080Fixer_SuperUltraCompact_op20_clamp.onnx",
        },
        "gmfss": {
            "pth": "gmfss-fortuna-union.zip",
        },
        "rife4.25-heavy": {
            "pth": "rife425_heavy.pth",
            "onnx_fp16": "rife425_heavy.pth",
            "onnx_fp32": "rife425_heavy.pth",
            "ncnn": None,
        },
        "rife_elexor": {
            "pth": "rife_elexor.pth",
            "onnx_fp16": "rife_elexor.pth",
            "onnx_fp32": "rife_elexor.pth",
            "ncnn": "rife-elexor-ncnn.zip",
        },
        "rife4.25-lite": {
            "pth": "rife425_lite.pth",
            "onnx_fp16": None,
            "onnx_fp32": None,
            "ncnn": None,
        },
        "rife4.25": {
            "pth": "rife425.pth",
            "onnx_fp16": "rife425_fp16_op21_slim.onnx",
            "onnx_fp32": "rife425_fp32_op21_slim.onnx",
            "ncnn": None,
        },
        "rife4.22-lite": {
            "pth": "rife422_lite.pth",
            "onnx_fp16": "rife4.22_lite_fp16_op21_slim.onnx",
            "onnx_fp32": "rife4.22_lite_fp32_op21_slim.onnx",
            "ncnn": "rife-v4.22-lite-ncnn.zip",
        },
        "rife4.20": {
            "pth": "rife420.pth",
            "onnx_fp16": "rife420_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx",
            "onnx_fp32": "rife420_v2_ensembleFalse_op20_fp32_clamp_onnxslim.onnx",
            "ncnn": "rife-v4.20-ncnn.zip",
        },
        "rife": {
            "pth": "rife422.pth",
            "onnx_fp16": "rife422_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx",
            "onnx_fp32": "rife422_v2_ensembleFalse_op20_clamp_onnxslim.onnx",
            "ncnn": "rife-v4.22-ncnn.zip",
        },
        "rife4.21": {
            "pth": "rife421.pth",
            "onnx_fp16": "rife421_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx",
            "onnx_fp32": "rife421_v2_ensembleFalse_op20_clamp_onnxslim.onnx",
            "ncnn": "rife-v4.21-ncnn.zip",
        },
        "rife4.18": {
            "pth": "rife418.pth",
            "onnx_fp16": "rife418_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx",
            "onnx_fp32": "rife418_v2_ensembleFalse_op20_clamp_onnxslim.onnx",
            "ncnn": "rife-v4.18-ncnn.zip",
        },
        "rife4.17": {
            "pth": "rife417.pth",
            "onnx_fp16": "rife417_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx",
            "onnx_fp32": "rife417_v2_ensembleFalse_op20_clamp_onnxslim.onnx",
            "ncnn": "rife-v4.17-ncnn.zip",
        },
        "rife4.15-lite": {
            "pth": "rife415_lite.pth",
            "onnx_fp16": "rife_v4.15_lite_fp16_op20_sim.onnx",
            "onnx_fp32": "rife_v4.15_lite_fp32_op20_sim.onnx",
            "ncnn": "rife-v4.15-lite-ncnn.zip",
        },
        "rife4.15": {
            "pth": "rife415.pth",
            "onnx_fp16": "rife415_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx",
            "onnx_fp32": "rife415_v2_ensembleFalse_op20_fp32_clamp_onnxslim.onnx",
            "ncnn": "rife-v4.15-ncnn.zip",
        },
        "rife4.6": {
            "pth": "rife46.pth",
            "onnx_fp16": "rife46_v2_ensembleFalse_op16_fp16_mlrt_sim.onnx",
            "onnx_fp32": "rife46_v2_ensembleFalse_op16_mlrt_sim.onnx",
            "ncnn": "rife-v4.6-ncnn.zip",
        },
        "rife4.16-lite": {
            "pth": "rife416_lite.pth",
            "onnx_fp16": None,
            "onnx_fp32": None,
            "ncnn": "rife-v4.16-lite-ncnn.zip",
        },
        "segment-tensorrt": {
            "onnx": "isnet_is.onnx",
        },
        "maxxvit-tensorrt": {
            "onnx_fp16": "maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx",
            "onnx_fp32": "maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_op17_onnxslim.onnx",
        },
        "shift_lpips": {
            "onnx_fp16": "sc_shift_lpips_alex_256px_CHW_6ch_clamp_op20_fp16_onnxslim.onnx",
            "onnx_fp32": "sc_shift_lpips_alex_256px_CHW_6ch_clamp_op20_onnxslim.onnx",
        },
        "differential-tensorrt": {
            "onnx": "scene_change_nilas.onnx",
        },
        "small_v2": {
            "pth": "depth_anything_v2_vits.pth",
        },
        "base_v2": {
            "pth": "depth_anything_v2_vitb.pth",
        },
        "large_v2": {
            "pth": "depth_anything_v2_vitl.pth",
        },
        "small_v2_directml": {
            "onnx_fp16": "depth_anything_v2_vits14_float16_slim.onnx",
            "onnx_fp32": "depth_anything_v2_vits14_float32_slim.onnx",
        },
        "small_v2_tensorrt": {
            "onnx_fp16": "depth_anything_v2_vits14_float16_slim.onnx",
            "onnx_fp32": "depth_anything_v2_vits14_float32_slim.onnx",
        },
        "base_v2_directml": {
            "onnx_fp16": "depth_anything_v2_vitb14_float16_slim.onnx",
            "onnx_fp32": "depth_anything_v2_vitb14_float32_slim.onnx",
        },
        "base_v2_tensorrt": {
            "onnx_fp16": "depth_anything_v2_vitb14_float16_slim.onnx",
            "onnx_fp32": "depth_anything_v2_vitb14_float32_slim.onnx",
        },
        "large_v2_directml": {
            "onnx_fp16": "depth_anything_v2_vitl14_float16_slim.onnx",
            "onnx_fp32": "depth_anything_v2_vitl14_float32_slim.onnx",
        },
        "large_v2_tensorrt": {
            "onnx_fp16": "depth_anything_v2_vitl14_float16_slim.onnx",
            "onnx_fp32": "depth_anything_v2_vitl14_float32_slim.onnx",
        },
    }

    key = model
    if model.endswith("-tensorrt") or model.endswith("-directml"):
        if "rife" in model:
            key = model.replace("-tensorrt", "").replace("-directml", "")
        else:
            key = model.replace("-tensorrt", "").replace("-directml", "")

    modelInfo = modelDict.get(model) or modelDict.get(key)

    if not modelInfo:
        raise ValueError(f"Model {model} not found.")

    if modelType == "pth":
        filename = modelInfo.get("pth")
        if not filename:
            raise ValueError(f"PTH model for {model} not found.")
        return filename

    elif modelType == "onnx":
        if "onnx_fp16" in modelInfo and "onnx_fp32" in modelInfo:
            if ensemble:
                if key.startswith("rife4.25") or key.startswith("rife4.22"):
                    print(
                        "Starting rife 4.21 Ensemble is no longer going to be supported."
                    )
                elif (
                    key.startswith("rife4.21")
                    or key.startswith("rife4.18")
                    or key.startswith("rife4.17")
                ):
                    print(
                        "Starting rife 4.21 Ensemble is no longer going to be supported."
                    )
            if half:
                filename = modelInfo.get("onnx_fp16")
            else:
                filename = modelInfo.get("onnx_fp32")
            if not filename:
                raise ValueError(f"ONNX model for {model} with half={half} not found.")
            return filename
        elif "onnx" in modelInfo:
            filename = modelInfo.get("onnx")
            if not filename:
                raise ValueError(f"ONNX model for {model} not found.")
            return filename
        else:
            raise ValueError(f"ONNX model type for {model} is not supported.")

    elif modelType == "ncnn":
        filename = modelInfo.get("ncnn")
        if filename is None:
            if key in ["rife4.25-heavy", "rife4.25-lite"]:
                raise ValueError("NCNN model not found.")
            else:
                pass
        if filename == "raise_error":
            raise ValueError("NCNN model not found.")
        if not filename:
            raise ValueError(f"NCNN model for {model} not found.")
        return filename

    else:
        raise ValueError(f"Unsupported model type: {modelType}")


def downloadAndLog(
    model: str, filename: str, download_url: str, folderPath: str, retries: int = 3
):
    import requests

    tempFolder = os.path.join(folderPath, "TEMP")
    os.makedirs(tempFolder, exist_ok=True)

    for attempt in range(retries):
        try:
            if os.path.exists(os.path.join(folderPath, filename)):
                toLog = f"{model.upper()} model already exists at: {os.path.join(folderPath, filename)}"
                logging.info(toLog)
                return os.path.join(folderPath, filename)

            toLog = f"Downloading {model.upper()} model... (Attempt {attempt + 1}/{retries})"
            logging.info(toLog)

            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            try:
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                total_size_in_mb = total_size_in_bytes / (
                    1024 * 1024
                )  # Convert bytes to MB
            except Exception as e:
                total_size_in_mb = 0  # If there's an error, default to 0 MB
                logging.error(e)

            tempFilePath = os.path.join(tempFolder, filename)

            with ProgressBarDownloadLogic(
                int(total_size_in_mb + 1),
                title=f"Downloading {model.upper()} model... (Attempt {attempt + 1}/{retries})",
            ) as bar:
                with open(tempFilePath, "wb") as file:
                    for data in response.iter_content(chunk_size=1024 * 1024):
                        file.write(data)
                        bar(int(len(data) / (1024 * 1024)))

            if filename.endswith(".zip"):
                import zipfile

                with zipfile.ZipFile(tempFilePath, "r") as zip_ref:
                    zip_ref.extractall(folderPath)
                os.remove(tempFilePath)
                filename = filename[:-4]
            else:
                os.rename(tempFilePath, os.path.join(folderPath, filename))

            os.rmdir(tempFolder)

            toLog = f"Downloaded {model.capitalize()} model to: {os.path.join(folderPath, filename)}"
            logging.info(toLog)
            print(green(toLog))

            return os.path.join(folderPath, filename)

        except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
            logging.error(f"Error during download: {e}")
            if os.path.exists(os.path.join(folderPath, filename)):
                os.remove(os.path.join(folderPath, filename))
            if attempt == retries - 1:
                raise

    return None


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

    elif model == "small_v2":
        fullUrl = f"{DEPTHV2URLSMALL}{filename}"
    elif model == "base_v2":
        fullUrl = f"{DEPTHV2URLBASE}{filename}"
    elif model == "large_v2":
        fullUrl = f"{DEPTHV2URLLARGE}{filename}"

    else:
        fullUrl = f"{TASURL}{filename}"

    return downloadAndLog(model, filename, fullUrl, folderPath)
