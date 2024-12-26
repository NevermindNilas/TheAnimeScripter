import os
import logging
from .coloredPrints import green, yellow
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
        "rife4.6-ncnn",
        "rife4.15-lite-ncnn",
        "rife4.16-lite-ncnn",
        "rife4.17-ncnn",
        "rife4.18-ncnn",
        "rife4.20-ncnn",
        "rife4.21-ncnn",
        "rife4.22-ncnn",
        "rife4.22-lite-ncnn",
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
        "flownets",
    ]


def modelsMap(
    model: str = "compact",
    upscaleFactor: int = 2,
    modelType="pth",
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
    """

    match model:
        case "flownets":
            return "flownets.pth"

        case "shufflespan" | "shufflespan-directml" | "shufflespan-tensorrt":
            if modelType == "pth":
                return "sudo_shuffle_span_10.5m.pth"
            else:
                if half:
                    return "sudo_shuffle_span_op20_10.5m_1080p_fp16_op21_slim.onnx"
                else:
                    return "sudo_shuffle_span_op20_10.5m_1080p_fp32_op21_slim.onnx"

        case "aniscale2" | "aniscale2-directml" | "aniscale2-tensorrt":
            if modelType == "pth":
                return "2x_AniScale2S_Compact_i8_60K.pth"
            else:
                if half:
                    return "2x_AniScale2S_Compact_i8_60K-fp16.onnx"
                else:
                    return "2x_AniScale2S_Compact_i8_60K-fp32.onnx"

        case "open-proteus" | "open-proteus-directml" | "open-proteus-tensorrt":
            if modelType == "pth":
                return "2x_OpenProteus_Compact_i2_70K.pth"
            else:
                if half:
                    return "2x_OpenProteus_Compact_i2_70K-fp16.onnx"
                else:
                    return "2x_OpenProteus_Compact_i2_70K-fp32.onnx"

        case "compact" | "compact-directml" | "compact-tensorrt":
            if modelType == "pth":
                return "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth"
            else:
                if half:
                    return "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k_clamp_fp16_op18_onnxslim.onnx"
                else:
                    return "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k_clamp_op18_onnxslim.onnx"

        case "ultracompact" | "ultracompact-directml" | "ultracompact-tensorrt":
            if modelType == "pth":
                return "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k.pth"
            else:
                if half:
                    return "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k_clamp_fp16_op18_onnxslim.onnx"
                else:
                    return "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k_clamp_op18_onnxslim.onnx"

        case (
            "superultracompact"
            | "superultracompact-directml"
            | "superultracompact-tensorrt"
        ):
            if modelType == "pth":
                return "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k.pth"
            else:
                if half:
                    return "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k_clamp_fp16_op18_onnxslim.1.onnx"

                else:
                    return "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k_clamp_op18_onnxslim.onnx"

        case "span" | "span-directml" | "span-tensorrt" | "span-ncnn":
            if modelType == "pth":
                return "2x_ModernSpanimationV2.pth"
            elif modelType == "onnx":
                if half:
                    return "2x_ModernSpanimationV2_fp16_op21_slim.onnx"
                else:
                    return "2x_ModernSpanimationV2_fp32_op21_slim.onnx"
            elif modelType == "ncnn":
                return "2x_modernspanimationv1.5-ncnn.zip"

        case (
            "shufflecugan"
            | "shufflecugan-directml"
            | "shufflecugan-tensorrt"
            | "shufflecugan-ncnn"
        ):
            if modelType == "pth":
                return "sudo_shuffle_cugan_9.584.969.pth"
            elif modelType == "onnx":
                if half:
                    return "sudo_shuffle_cugan_fp16_op18_clamped.onnx"
                else:
                    return "sudo_shuffle_cugan_op18_clamped.onnx"
            elif modelType == "ncnn":
                return "2xsudo_shuffle_cugan-ncnn.zip"

        case "segment":
            return "isnetis.ckpt"

        case "scunet":
            return "scunet_color_real_psnr.pth"

        case "dpir":
            return "drunet_deblocking_color.pth"

        case "real-plksr":
            return "1xDeJPG_realplksr_otf.pth"

        case "nafnet":
            return "NAFNet-GoPro-width64.pth"

        case "anime1080fixer" | "anime1080fixer-tensorrt" | "anime1080fixer-directml":
            if modelType == "pth":
                return "1x_Anime1080Fixer_SuperUltraCompact.pth"
            elif modelType == "onnx":
                if half:
                    return "1x_Anime1080Fixer_SuperUltraCompact_op20_fp16_clamp.onnx"
                else:
                    return "1x_Anime1080Fixer_SuperUltraCompact_op20_clamp.onnx"

        case "gmfss":
            return "gmfss-fortuna-union.zip"

        case "rife4.25-heavy" | "rife4.25-heavy-tensorrt" | "rife4.25-heavy-ncnn":
            if modelType == "pth":
                return "rife425_heavy.pth"
            elif modelType == "onnx":
                return "rife425_heavy.pth"
            elif modelType == "ncnn":
                raise ValueError("NCNN model not found.")

        case "rife_elexor" | "rife_elexor-tensorrt" | "rife_elexor-ncnn":
            if modelType == "pth":
                return "rife_elexor.pth"
            elif modelType == "onnx":
                return "rife_elexor.pth"
            elif modelType == "ncnn":
                raise ValueError("NCNN model not found.")

        case "rife4.25-lite" | "rife4.25-lite-tensorrt" | "rife4.25-lite-ncnn":
            if modelType == "pth":
                return "rife425_lite.pth"
            elif modelType == "onnx":
                pass
            elif modelType == "ncnn":
                raise ValueError("NCNN model not found.")

        case "rife4.25" | "rife4.25-tensorrt" | "rife4.25-ncnn":
            if modelType == "pth":
                return "rife425.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        print(
                            "Starting rife 4.21 Ensemble is no longer going to be supported."
                        )
                    else:
                        return "rife425_fp16_op21_slim.onnx"
                else:
                    if ensemble:
                        print(
                            "Starting rife 4.21 Ensemble is no longer going to be supported."
                        )
                    else:
                        return "rife425_fp32_op21_slim.onnx"

            elif modelType == "ncnn":
                pass

        case "rife4.22-lite" | "rife4.22-lite-tensorrt" | "rife4.22-lite-ncnn":
            if modelType == "pth":
                return "rife422_lite.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        print(
                            "Starting rife 4.21 Ensemble is no longer going to be supported."
                        )
                        return "rife4.22_lite_fp16_op21_slim.onnx"
                    else:
                        return "rife4.22_lite_fp16_op21_slim.onnx"
                else:
                    if ensemble:
                        print(
                            "Starting rife 4.21 Ensemble is no longer going to be supported."
                        )
                        return "rife4.22_lite_fp32_op21_slim.onnx"
                    else:
                        return "rife4.22_lite_fp32_op21_slim.onnx"
            elif modelType == "ncnn":
                if ensemble:
                    print(
                        yellow(
                            "Starting rife 4.21 Ensemble is no longer going to be supported."
                        )
                    )
                    return "rife-v4.22-lite-ensemble-ncnn.zip"
                else:
                    return "rife-v4.22-lite-ncnn.zip"

        case "rife4.20" | "rife4.20-tensorrt" | "rife4.20-ncnn":
            if modelType == "pth":
                return "rife420.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        return "rife420_v2_ensembleTrue_op20_fp16_clamp_onnxslim.onnx"
                    else:
                        return "rife420_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
                else:
                    if ensemble:
                        return "rife420_v2_ensembleTrue_op20_clamp_onnxslim.onnx"
                    else:
                        return "rife420_v2_ensembleFalse_op20_clamp_onnxslim.onnx"
            elif modelType == "ncnn":
                if ensemble:
                    return "rife-v4.20-ensemble-ncnn.zip"
                else:
                    return "rife-v4.20-ncnn.zip"

        case "rife" | "rife4.22" | "rife4.22-tensorrt" | "rife4.22-ncnn":
            if modelType == "pth":
                return "rife422.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        print(
                            yellow(
                                "Starting rife 4.21 Ensemble is no longer going to be supported."
                            )
                        )
                        return "rife422_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
                    else:
                        return "rife422_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
                else:
                    if ensemble:
                        print(
                            yellow(
                                "Starting rife 4.21 Ensemble is no longer going to be supported."
                            )
                        )
                        return "rife422_v2_ensembleFalse_op20_clamp_onnxslim.onnx"
                    else:
                        return "rife422_v2_ensembleFalse_op20_clamp_onnxslim.onnx"
            elif modelType == "ncnn":
                if ensemble:
                    print(
                        yellow(
                            "Starting rife 4.21 Ensemble is no longer going to be supported."
                        )
                    )
                    return "rife-v4.22-ncnn.zip"
                else:
                    return "rife-v4.22-ncnn.zip"

        case "rife4.21" | "rife4.21-tensorrt" | "rife4.21-ncnn":
            if modelType == "pth":
                return "rife421.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        print(
                            yellow(
                                "Starting rife 4.21 Ensemble is no longer going to be supported."
                            )
                        )
                        return "rife421_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
                    else:
                        return "rife421_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
                else:
                    if ensemble:
                        print(
                            yellow(
                                "Starting rife 4.21 Ensemble is no longer going to be supported."
                            )
                        )
                        return "rife421_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
                    else:
                        return "rife421_v2_ensembleFalse_op20_clamp_onnxslim.onnx"
            elif modelType == "ncnn":
                if ensemble:
                    print(
                        yellow(
                            "Starting rife 4.21 Ensemble is no longer going to be supported."
                        )
                    )
                    return "rife-v4.21-ncnn.zip"
                else:
                    return "rife-v4.21-ncnn.zip"

        case "rife4.18" | "rife4.18-tensorrt" | "rife4.18-ncnn":
            if modelType == "pth":
                return "rife418.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        return "rife418_v2_ensembleTrue_op20_fp16_clamp_onnxslim.onnx"
                    else:
                        return "rife418_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
                else:
                    if ensemble:
                        return "rife418_v2_ensembleTrue_op20_clamp_onnxslim.onnx"
                    else:
                        return "rife418_v2_ensembleFalse_op20_clamp_onnxslim.onnx"
            elif modelType == "ncnn":
                if ensemble:
                    return "rife-v4.18-ensemble-ncnn.zip"
                else:
                    return "rife-v4.18-ncnn.zip"

        case "rife4.17" | "rife4.17-tensorrt" | "rife4.17-ncnn":
            if modelType == "pth":
                return "rife417.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        return "rife417_v2_ensembleTrue_op20_fp16_clamp_onnxslim.onnx"
                    else:
                        return "rife417_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
                else:
                    if ensemble:
                        return "rife417_v2_ensembleTrue_op20_clamp_onnxslim.onnx"
                    else:
                        return "rife417_v2_ensembleFalse_op20_clamp_onnxslim.onnx"
            elif modelType == "ncnn":
                if ensemble:
                    return "rife-v4.17-ensemble-ncnn.zip"
                else:
                    return "rife-v4.17-ncnn.zip"

        case "rife4.15-lite" | "rife4.15-lite-tensorrt" | "rife4.15-lite-ncnn":
            if modelType == "pth":
                return "rife415_lite.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        return "rife_v4.15_lite_ensemble_fp16_op20_sim.onnx"
                    else:
                        return "rife_v4.15_lite_fp16_op20_sim.onnx"
                else:
                    if ensemble:
                        return "rife_v4.15_lite_ensemble_fp32_op20_sim.onnx"
                    else:
                        return "rife_v4.15_lite_fp32_op20_sim.onnx"
            elif modelType == "ncnn":
                if ensemble:
                    return "rife-v4.15-lite-ensenmble-ncnn.zip"
                else:
                    return "rife-v4.15-lite-ncnn.zip"

        case "rife4.15" | "rife4.15-tensorrt" | "rife4.15-ncnn":
            if modelType == "pth":
                return "rife415.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        return "rife415_v2_ensembleTrue_op20_fp16_clamp_onnxslim.onnx"
                    else:
                        return "rife415_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
                else:
                    if ensemble:
                        return "rife415_v2_ensembleTrue_op20_clamp_onnxslim.onnx"
                    else:
                        return "rife415_v2_ensembleFalse_op20_clamp_onnxslim.onnx"

            elif modelType == "ncnn":
                if ensemble:
                    return "rife-v4.15-ensemble-ncnn.zip"
                else:
                    return "rife-v4.15-ncnn.zip"

        case "rife4.6" | "rife4.6-tensorrt" | "rife4.6-ncnn":
            if modelType == "pth":
                return "rife46.pth"
            elif modelType == "onnx":
                if half:
                    if ensemble:
                        return "rife46_v2_ensembleTrue_op16_fp16_mlrt_sim.onnx"
                    else:
                        return "rife46_v2_ensembleFalse_op16_fp16_mlrt_sim.onnx"
                else:
                    if ensemble:
                        return "rife46_v2_ensembleTrue_op16_mlrt_sim.onnx"
                    else:
                        return "rife46_v2_ensembleFalse_op16_mlrt_sim.onnx"

            elif modelType == "ncnn":
                return "rife-v4.6-ncnn.zip"

        case "segment-tensorrt" | "segment-directml":
            return "isnet_is.onnx"

        case "maxxvit-tensorrt" | "maxxvit-directml":
            if half:
                return "maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx"
            else:
                return "maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_op17_onnxslim.onnx"

        case "shift_lpips-tensorrt" | "shift_lpips-directml":
            if half:
                return "sc_shift_lpips_alex_256px_CHW_6ch_clamp_op20_fp16_onnxslim.onnx"
            else:
                return "sc_shift_lpips_alex_256px_CHW_6ch_clamp_op20_onnxslim.onnx"

        case "differential-tensorrt":
            return "scene_change_nilas.onnx"

        case "small_v2":
            return "depth_anything_v2_vits.pth"

        case "base_v2":
            return "depth_anything_v2_vitb.pth"

        case "large_v2":
            return "depth_anything_v2_vitl.pth"

        case "small_v2-directml" | "small_v2-tensorrt":
            if half:
                return "depth_anything_v2_vits14_float16_slim.onnx"
            else:
                return "depth_anything_v2_vits14_float32_slim.onnx"

        case "base_v2-directml" | "base_v2-tensorrt":
            if half:
                return "depth_anything_v2_vitb14_float16_slim.onnx"
            else:
                return "depth_anything_v2_vitb14_float32_slim.onnx"

        case "large_v2-directml" | "large_v2-tensorrt":
            if half:
                return "depth_anything_v2_vitl14_float16_slim.onnx"
            else:
                return "depth_anything_v2_vitl14_float32_slim.onnx"

        case _:
            raise ValueError(f"Model {model} not found.")


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
