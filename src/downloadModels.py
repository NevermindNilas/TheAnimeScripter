import os
import logging
import requests
from tqdm import tqdm

dirPath = os.path.dirname(__file__)
weightsDir = os.path.join(dirPath, "weights")
TASURL = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/"
DEPTHURL = (
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/"
)
SUDOURL = "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"


def modelsList() -> list[str]:
    return [
        "compact",
        "ultracompact",
        "superultracompact",
        "span",
        "omnisr",
        "shufflecugan",
        "cugan",
        "segment",
        "segment-tensorrt",
        "scunet",
        "dpir",
        "realesrgan",
        "nafnet",
        "span-denoise",
        "gmfss",
        "rife",
        "rife4.6",
        "rife4.15",
        "rife4.15-lite",
        "rife4.16-lite",
        "rife4.17",
        "vits",
        "vitb",
        "vitl",
        "sam-vitb",
        "sam-vitl",
        "sam-vith",
        "cugan-directml",
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

        case "omnisr":
            return "2xHFA2kOmniSR.pth"

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

        case "cugan" | "cugan-directml":
            if modelType == "pth":
                return "2xHFA2kReal-CUGAN.pth"
            else:
                if half:
                    return "2xHFA2kReal-CUGAN-fp16.onnx"
                else:
                    return "2xHFA2kReal-CUGAN-fp32.onnx"

        case "segment":
            return "isnetis.ckpt"

        case "scunet":
            return "scunet_color_real_psnr.pth"

        case "dpir":
            return "drunet_deblocking_color.pth"

        case "realesrgan":
            return "2xHFA2kShallowESRGAN.pth"

        case "nafnet":
            return "NAFNet-GoPro-width64.pth"

        case "span-denoise":
            return "1x_span_anime_pretrain.pth"

        case "gmfss":
            return "gmfss-fortuna-union.zip"

        case "rife" | "rife4.17":
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

        case "sam-vitb":
            return "sam_vit_b_01ec64.pth"

        case "sam-vitl":
            return "sam_vit_l_0b3195.pth"

        case "sam-vith":
            return "sam_vit_h_4b8939.pth"

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
        
        case "small-tensorrt":
            if half:
                return "depth_anything_vits14_float16_slim.onnx"
            else:
                return "depth_anything_vits14_float32_slim.onnx"
        
        case "base-tensorrt":
            if half:
                return "depth_anything_vitb14_float16_slim.onnx"
            else:
                return "depth_anything_vitb14_float32_slim.onnx"
            
        case "large-tensorrt":
            if half:
                return "depth_anything_vitl14_float16_slim.onnx"
            else:
                return "depth_anything_vitl14_float32_slim.onnx"
        
        case "segment-tensorrt":
            return "isnet_is.onnx"

        case "rife4.6-tensorrt":
            if half:
                if ensemble:
                    return "rife_v4.6_ensemble_fp16_op20_sim.onnx"
                else:
                    return "rife_v4.6_fp16_op20_sim.onnx"
            else:
                if ensemble:
                    return "rife_v4.6_ensemble_op20_sim.onnx"
                else:
                    return "rife_v4.6_op20_sim.onnx"
                
        case "rife4.17-tensorrt":
            if half:
                if ensemble:
                    return "rife417_v2_ensembleTrue_op20_clamp_fp16_onnxslim.onnx"
                else:
                    return "rife417_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx"
            else:
                if ensemble:
                    return "rife417_v2_ensembleTrue_op20_clamp_onnxslim.onnx"
                else:
                    return "rife417_v2_ensembleFalse_op20_clamp_onnxslim.onnx"
        case _:
            raise ValueError(f"Model {model} not found.")


def downloadAndLog(model: str, filename: str, download_url: str, folderPath: str):
    if os.path.exists(os.path.join(folderPath, filename)):
        toLog = f"{model.upper()} model already exists at: {os.path.join(folderPath, filename)}"
        logging.info(toLog)
        return os.path.join(folderPath, filename)

    toLog = f"Downloading {model.upper()} model..."
    logging.info(toLog)
    print(toLog)

    response = requests.get(download_url, stream=True)
    
    try:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        if total_size_in_bytes == 0:
            total_size_in_bytes = None
    except Exception as e:
        total_size_in_bytes = None
        logging.error(e)

    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, colour="green"
    )

    with open(os.path.join(folderPath, filename), "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if filename.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(os.path.join(folderPath, filename), "r") as zip_ref:
            zip_ref.extractall(folderPath)
        os.remove(os.path.join(folderPath, filename))
        filename = filename[:-4]

    toLog = f"Downloaded {model.upper()} model to: {os.path.join(folderPath, filename)}"

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
    elif model in ["rife4.15-tensorrt", "rife4.17-tensorrt"]:
        fullUrl = f"{SUDOURL}{filename}"
    else:
        fullUrl = f"{TASURL}{filename}"

    return downloadAndLog(model, filename, fullUrl, folderPath)
