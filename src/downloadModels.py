import os
import logging
import requests
from alive_progress import alive_bar

if os.name == "nt":
    appdata = os.getenv("APPDATA")
    mainPath = os.path.join(appdata, "TheAnimeScripter")

    if not os.path.exists(mainPath):
        os.makedirs(mainPath)

    weightsDir = os.path.join(mainPath, "weights")
else:
    dirPath = os.path.dirname(__file__)
    weightsDir = os.path.join(dirPath, "weights")

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
        "gmfss",
        "rife",
        "rife4.6",
        "rife4.15",
        "rife4.15-lite",
        "rife4.16-lite",
        "rife4.17",
        "rife4.18",
        "vits",
        "vitb",
        "vitl",
        "shufflecugan-directml",
        "compact-directml",
        "ultracompact-directml",
        "superultracompact-directml",
        "span-directml",
        "shufflecugan-tensorrt",
        "compact-tensorrt",
        "ultracompact-tensorrt",
        "superultracompact-tensorrt",
        "span-tensorrt",
        "rife4.6-tensorrt",
        "rife4.15-tensorrt",
        "rife4.15-lite-tensorrt",
        "rife4.17-tensorrt",
        "rife-v4.15-ncnn",
        "rife-v4.6-ncnn",
        "rife-v4.15-lite-ncnn",
        "rife-v4.16-lite-ncnn",
        "rife-v4.17-ncnn",
        "rife-v4.18-ncnn",
        "scenechange",
        "small-tensorrt",
        "base-tensorrt",
        "large-tensorrt",
        "small-directml",
        "base-directml",
        "large-directml",
        "small_v2",
        "base_v2",
        "large_v2",
        "small_v2-directml",
        "base_v2-directml",
        "large_v2-directml",
        "small_v2-tensorrt",
        "base_v2-tensorrt",
        "large_v2-tensorrt",
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
        case "compact" | "compact-directml":
            if modelType == "pth":
                return "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth"
            else:
                if half:
                    return "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k-fp16-sim.onnx"
                else:
                    return "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k-fp32-sim.onnx"

        case "ultracompact" | "ultracompact-directml":
            if modelType == "pth":
                return "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k.pth"
            else:
                if half:
                    return "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k-fp16-sim.onnx"
                else:
                    return "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k-fp32-sim.onnx"

        case "superultracompact" | "superultracompact-directml":
            if modelType == "pth":
                return "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k.pth"
            else:
                if half:
                    return (
                        "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k-fp16-sim.onnx"
                    )
                else:
                    return (
                        "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k-fp32-sim.onnx"
                    )

        case "compact-tensorrt":
            if half:
                return "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k_clamp_fp16_op18_onnxslim.onnx"
            else:
                return (
                    "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k_clamp_op18_onnxslim.onnx"
                )

        case "ultracompact-tensorrt":
            if half:
                return "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k_clamp_fp16_op18_onnxslim.onnx"
            else:
                return "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k_clamp_op18_onnxslim.onnx"

        case "superultracompact-tensorrt":
            if half:
                return "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k_clamp_fp16_op18_onnxslim.1.onnx"
            else:
                return "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k_clamp_op18_onnxslim.onnx"

        case "span" | "span-directml":
            if modelType == "pth":
                return "2x_ModernSpanimationV1.pth"
            else:
                if half:
                    return "2x_ModernSpanimationV1_fp16_op17.onnx"
                else:
                    return "2x_ModernSpanimationV1_fp32_op17.onnx"

        case "span-tensorrt":
            if half:
                return "2x_ModernSpanimationV1_clamp_fp16_op19_onnxslim.onnx"
            else:
                return "2x_ModernSpanimationV1_clamp_op19_onnxslim.onnx"

        case "shufflecugan" | "shufflecugan-directml":
            if modelType == "pth":
                return "sudo_shuffle_cugan_9.584.969.pth"
            else:
                if half:
                    return "sudo_shuffle_cugan_9.584.969-fp16.onnx"
                else:
                    return "sudo_shuffle_cugan_9.584.969-fp32.onnx"

        case "shufflecugan-tensorrt":
            if half:
                return "sudo_shuffle_cugan_fp16_op18_clamped.onnx"
            else:
                return "sudo_shuffle_cugan_op18_clamped.onnx"

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

        case "gmfss":
            return "gmfss-fortuna-union.zip"
        
        case "rife" | "rife4.20":
            return "rife420.pth"

        case "rife4.18":
            return "rife418.pth"

        case "rife4.17":
            return "rife417.pth"

        case "rife4.15":
            return "rife415.pth"

        case "rife4.15-lite":
            return "rife415_lite.pth"

        case "rife4.6":
            return "rife46.pth"

        case "rife4.15-tensorrt":
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

        case "rife4.15-lite-tensorrt":
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

        case "rife4.16-lite":
            return "rife416_lite.pth"

        case "vits":
            return "depth_anything_vits14.pth"

        case "vitb":
            return "depth_anything_vitb14.pth"

        case "vitl":
            return "depth_anything_vitl14.pth"

        case "rife-v4.18-ncnn":
            if ensemble:
                return "rife-v4.18-ensemble-ncnn.zip"
            else:
                return "rife-v4.18-ncnn.zip"

        case "rife-v4.17-ncnn":
            if ensemble:
                return "rife-v4.17-ensemble-ncnn.zip"
            else:
                return "rife-v4.17-ncnn.zip"

        case "rife-v4.16-lite-ncnn":
            if ensemble:
                return "rife-v4.16-lite-ensemble-ncnn.zip"
            else:
                return "rife-v4.16-lite-ncnn.zip"

        case "rife-v4.15-ncnn":
            if ensemble:
                return "rife-v4.15-ensemble-ncnn.zip"
            else:
                return "rife-v4.15-ncnn.zip"

        case "rife-v4.6-ncnn":
            if ensemble:
                return "rife-v4.6-ensemble-ncnn.zip"
            else:
                return "rife-v4.6-ncnn.zip"

        case "rife-v4.15-lite-ncnn":
            if ensemble:
                return "rife-v4.15-lite-ensenmble-ncnn.zip"
            else:
                return "rife-v4.15-lite-ncnn.zip"

        case "small-tensorrt" | "small-directml":
            if half:
                return "depth_anything_vits14_float16_slim.onnx"
            else:
                return "depth_anything_vits14_float32_slim.onnx"

        case "base-tensorrt" | "base-directml":
            if half:
                return "depth_anything_vitb14_float16_slim.onnx"
            else:
                return "depth_anything_vitb14_float32_slim.onnx"

        case "large-tensorrt" | "large-directml":
            if half:
                return "depth_anything_vitl14_float16_slim.onnx"
            else:
                return "depth_anything_vitl14_float32_slim.onnx"

        case "segment-tensorrt" | "segment-directml":
            return "isnet_is.onnx"

        case "rife4.6-tensorrt":
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

        case "rife4.18-tensorrt":
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

        case "rife4.17-tensorrt":
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

        case "scenechange":
            if half:
                return "maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx"
            else:
                return "maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_op17_onnxslim.onnx"

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


def downloadAndLog(model: str, filename: str, download_url: str, folderPath: str):
    if os.path.exists(os.path.join(folderPath, filename)):
        toLog = f"{model.upper()} model already exists at: {os.path.join(folderPath, filename)}"
        logging.info(toLog)
        return os.path.join(folderPath, filename)

    toLog = f"Downloading {model.upper()} model..."
    logging.info(toLog)

    response = requests.get(download_url, stream=True)

    try:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        total_size_in_mb = total_size_in_bytes / (1024 * 1024)  # Convert bytes to MB
    except Exception as e:
        total_size_in_mb = 0  # If there's an error, default to 0 MB
        logging.error(e)

    with alive_bar(
        int(total_size_in_mb + 1),  # Hacky but it works
        title=f"Downloading {model.capitalize()} model",
        bar="smooth",
        unit="MB",
        spinner=True,
        enrich_print=False,
        receipt=True,
        monitor=True,
        elapsed=True,
        stats=False,
        dual_line=False,
        force_tty=True,
    ) as bar:
        with open(os.path.join(folderPath, filename), "wb") as file:
            for data in response.iter_content(chunk_size=1024 * 1024):
                file.write(data)
                bar(int(len(data) / (1024 * 1024)))

    if filename.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(os.path.join(folderPath, filename), "r") as zip_ref:
            zip_ref.extractall(folderPath)
        os.remove(os.path.join(folderPath, filename))
        filename = filename[:-4]

    toLog = f"Downloaded {model.capitalize()} model to: {os.path.join(folderPath, filename)}"
    logging.info(toLog)
    print(toLog)

    return os.path.join(folderPath, filename)


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
    folderPath = os.path.join(weightsDir, model)
    os.makedirs(folderPath, exist_ok=True)

    if model in ["vits", "vitb", "vitl"]:
        fullUrl = f"{DEPTHURL}{filename}"
    elif model in [
        "rife4.18-tensorrt",
        "rife4.15-tensorrt",
        "rife4.17-tensorrt",
        "rife4.6-tensorrt",
        "scenechange",
    ]:
        fullUrl = f"{SUDOURL}{filename}"

    elif model == "small_v2":
        fullUrl = f"{DEPTHV2URLSMALL}{filename}"
    elif model == "base_v2":
        fullUrl = f"{DEPTHV2URLBASE}{filename}"
    elif model == "large_v2":
        fullUrl = f"{DEPTHV2URLLARGE}{filename}"

    else:
        fullUrl = f"{TASURL}{filename}"

    return downloadAndLog(model, filename, fullUrl, folderPath)
