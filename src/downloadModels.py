import wget
import os
import logging

dirPath = os.path.dirname(__file__)
weightsDir = os.path.join(dirPath, "weights")
url = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/"
cuganUrl = (
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"
)
depthURL = (
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/"
)


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
        "rife4.16-lite",
        "apisr",
        "vits",
        "vitb",
        "vitl",
    ]


def modelsMap(
    model,
    upscaleFactor: int = 2,
    cuganKind: str = "",
    modelType="pth",
    half: bool = True,
) -> str:
    """
    Maps the model to the corresponding filename.

    Args:
        model: The model to map.
        upscaleFactor: The upscale factor.
        cuganKind: The cugan kind.
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

        case "shufflecugan":
            return "sudo_shuffle_cugan_9.584.969.pth"

        case "cugan":
            return f"cugan_up{upscaleFactor}x-latest-{cuganKind}.pth"

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

        case "rife" | "rife4.15":
            return "rife415.pth"

        case "rife4.14":
            return "rife414.pth"

        case "rife4.6":
            return "rife46.pth"

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

        case _:
            raise ValueError(f"Model {model} not found.")


def downloadAndLog(model: str, filename: str, download_url: str, folderPath: str):
    if os.path.exists(os.path.join(folderPath, filename)):
        logging.info(
            f"{model.upper()} model already exists at: {os.path.join(folderPath, filename)}"
        )
        print(f"Found {model.upper()} at: {os.path.join(folderPath, filename)}")
        return os.path.join(folderPath, filename)

    print(f"Downloading {model.upper()} model...\n")
    logging.info(f"Downloading {model.upper()} model...")

    wget.download(download_url, out=os.path.join(folderPath, filename))

    if filename.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(os.path.join(folderPath, filename), "r") as zip_ref:
            zip_ref.extractall(folderPath)
        os.remove(os.path.join(folderPath, filename))
        filename = filename[:-4]

    logging.info(
        f"Downloaded {model.upper()} model to: {os.path.join(folderPath, filename)}"
    )
    print(
        f"\nDownloaded {model.upper()} model to: {os.path.join(folderPath, filename)}"
    )

    return os.path.join(folderPath, filename)


def downloadModels(
    model: str = None,
    cuganKind: str = None,
    upscaleFactor: int = 2,
    modelType: str = "pth",
    half: bool = True,
) -> str:
    """
    Downloads the model.

    Args:
        model: The model to download.
        cuganKind: The cugan kind.
        upscaleFactor: The upscale factor.
        modelType: The model type.
        half: Whether to use half precision or not.
    """
    os.makedirs(weightsDir, exist_ok=True)

    filename = modelsMap(model, upscaleFactor, cuganKind, modelType, half)

    match model:
        case "cugan" | "shufflecugan":
            cuganFolderPath = os.path.join(weightsDir, model)
            os.makedirs(cuganFolderPath, exist_ok=True)
            fullUrl = f"{url if model == 'shufflecugan' else cuganUrl}{filename}"
            return downloadAndLog(model, filename, fullUrl, cuganFolderPath)

        case (
            "compact"
            | "ultracompact"
            | "superultracompact"
            | "compact-directml"
            | "ultracompact-directml"
            | "superultracompact-directml"
        ):
            compactFolderPath = os.path.join(weightsDir, model)
            os.makedirs(compactFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, compactFolderPath)

        case "span" | "span-directml":
            spanFolderPath = os.path.join(weightsDir, model)
            os.makedirs(spanFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, spanFolderPath)

        case "omnisr":
            omnisrFolderPath = os.path.join(weightsDir, "omnisr")
            os.makedirs(omnisrFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, omnisrFolderPath)

        case "gmfss":
            gmfssFolderPath = os.path.join(weightsDir, "gmfss")
            os.makedirs(gmfssFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, gmfssFolderPath)

        case "rife" | "rife4.14" | "rife4.6" | "rife4.15" | "rife4.16-lite":
            rifeFolderPath = os.path.join(weightsDir, "rife")
            os.makedirs(rifeFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, rifeFolderPath)

        case "segment":
            segmentFolderPath = os.path.join(weightsDir, "segment")
            os.makedirs(segmentFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, segmentFolderPath)

        case "realesrgan":
            realesrganFolderPath = os.path.join(weightsDir, "realesrgan")
            os.makedirs(realesrganFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, realesrganFolderPath)

        case "scunet":
            scunetFolderPath = os.path.join(weightsDir, "scunet")
            os.makedirs(scunetFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, scunetFolderPath)

        case "nafnet":
            nafnetFolderPath = os.path.join(weightsDir, "nafnet")
            os.makedirs(nafnetFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, nafnetFolderPath)

        case "span-denoise":
            spanFolderPath = os.path.join(weightsDir, "span-denoise")
            os.makedirs(spanFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, spanFolderPath)

        case "dpir":
            dpirFolderPath = os.path.join(weightsDir, "dpir")
            os.makedirs(dpirFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, dpirFolderPath)

        case "apisr":
            apisrFolderPath = os.path.join(weightsDir, "apisr")
            os.makedirs(apisrFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, apisrFolderPath)

        case "vits":
            vitsFolderPath = os.path.join(weightsDir, "vits")
            os.makedirs(vitsFolderPath, exist_ok=True)
            fullUrl = f"{depthURL}{filename}"
            return downloadAndLog(model, filename, fullUrl, vitsFolderPath)

        case "vitb":
            vitbFolderPath = os.path.join(weightsDir, "vitb")
            os.makedirs(vitbFolderPath, exist_ok=True)
            fullUrl = f"{depthURL}{filename}"
            return downloadAndLog(model, filename, fullUrl, vitbFolderPath)

        case "vitl":
            vitlFolderPath = os.path.join(weightsDir, "vitl")
            os.makedirs(vitlFolderPath, exist_ok=True)
            fullUrl = f"{depthURL}{filename}"
            return downloadAndLog(model, filename, fullUrl, vitlFolderPath)

        case _:
            print(f"Model {model} not found.")
            return None
