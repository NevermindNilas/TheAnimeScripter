import os
import logging
import requests
from tqdm import tqdm

dirPath = os.path.dirname(__file__)
weightsDir = os.path.join(dirPath, "weights")
url = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/"
DEPTHURL = (
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/"
)
SEGMENTURL = "https://dl.fbaipublicfiles.com/segment_anything/"  # VITH is well over 2GB and instead of hosting on Github I will just use the official repo


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
        "scunet",
        "dpir",
        "realesrgan",
        "nafnet",
        "span-denoise",
        "gmfss",
        "rife",
        "rife4.14",
        "rife4.6",
        "rife4.15",
        "rife4.15-lite",
        "rife4.16-lite",
        "apisr",
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
        "rife-directml",
        "rife4.6-directml",
        "rife4.14-directml",
        "rife4.15-directml",
        "rife4.15-lite-directml",
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

        case "span" | "span-directml":
            if modelType == "pth":
                return "2x_ModernSpanimationV1.pth"
            else:
                if half:
                    return "2x_ModernSpanimationV1_fp16_op17.onnx"
                else:
                    return "2x_ModernSpanimationV1_fp32_op17.onnx"

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

        case "rife" | "rife4.15" | "rife-directml" | "rife4.15-directml":
            if modelType == "pth":
                return "rife415.pth"
            else:
                if half:
                    return "rife415-sim_fp16.onnx"
                else:
                    raise NotImplementedError  # "rife415-sim_fp32.onnx"

        case "rife4.15-lite" | "rife4.15-lite-directml":
            if modelType == "pth":
                return "rife415_lite.pth"
            else:
                if half:
                    return "rife415_lite-sim_fp16.onnx"
                else:
                    raise NotImplementedError  # "rife415_lite-fp32-sim.onnx"

        case "rife4.14" | "rife4.14-directml":
            if modelType == "pth":
                return "rife414.pth"
            else:
                if half:
                    return "rife414-sim_fp16.onnx"
                else:
                    raise NotImplementedError  # "rife414-fp32-sim.onnx"

        case "rife4.6" | "rife4.6-directml":
            if modelType == "pth":
                return "rife46.pth"
            else:
                if half:
                    return "rife46-sim_fp16.onnx"
                else:
                    raise NotImplementedError  # "rife46-fp32-sim.onnx"
                
        case "rife4.6-tensorrt":
            if half:
                if ensemble:
                    return "rife46_ensembleTrue_op18_fp16_clamp_sim.onnx"
                else:
                    return "rife46_ensembleFalse_op18_fp16_clamp_sim.onnx"
            else:
                if ensemble:
                    return "rife46_ensembleTrue_op18_clamp_sim.onnx"
                else:
                    return "rife46_ensembleFalse_op18_clamp_sim.onnx"
        
        case "rife4.14-tensorrt":
            if half:
                if ensemble:
                    return "rife414_ensembleTrue_op18_fp16_clamp_sim.onnx"
                else:
                    return "rife414_ensembleFalse_op18_fp16_clamp_sim.onnx"
            else:
                if ensemble:
                    return "rife414_ensembleTrue_op18_clamp_sim.onnx"
                else:
                    return "rife414_ensembleFalse_op18_clamp_sim.onnx"
                
        case "rife4.15-tensorrt":
            if half:
                if ensemble:
                    return "rife415_ensembleTrue_op19_fp16_clamp_sim.onnx"
                else:
                    return "rife415_ensembleFalse_op19_fp16_clamp_sim.onnx"
            else:
                if ensemble:
                    return "rife415_ensembleTrue_op19_clamp_sim.onnx"
                else:
                    return "rife415_ensembleFalse_op19_clamp_sim.onnx"
                
        case "rife4.15-lite-tensorrt":
            raise NotImplementedError

        case "rife4.16-lite":
            return "rife416_lite.pth"

        case "apisr":
            if upscaleFactor == 2:
                return "2x_APISR_RRDB_GAN_generator.pth"
            elif upscaleFactor == 4:
                return "4x_APISR_RRDB_GAN_generator.pth"

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
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, colour="green"
    )

    with open(os.path.join(folderPath, filename), "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

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
    model: str = None, upscaleFactor: int = 2, modelType: str = "pth", half: bool = True
) -> str:
    """
    Downloads the model.
    """
    os.makedirs(weightsDir, exist_ok=True)

    filename = modelsMap(model, upscaleFactor, modelType, half)
    folderPath = os.path.join(weightsDir, model)
    os.makedirs(folderPath, exist_ok=True)

    if model in ["vits", "vitb", "vitl"]:
        fullUrl = f"{DEPTHURL}{filename}"
    elif model in ["vit_h", "vit_l", "vit_b"]:
        fullUrl = f"{SEGMENTURL}{filename}"
    else:
        fullUrl = f"{url}{filename}"

    return downloadAndLog(model, filename, fullUrl, folderPath)
