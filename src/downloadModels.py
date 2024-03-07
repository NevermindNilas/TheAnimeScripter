import wget
import os
import logging

dirPath = os.path.dirname(__file__)
weightsDir = os.path.join(dirPath, "weights")
url = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/"
cuganUrl = (
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"
)


def modelsMap(model, upscaleFactor: int = 2, cuganKind: str = "") -> str:
    match model:
        case "compact":
            return "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth"

        case "ultracompact":
            return "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k.pth"

        case "superultracompact":
            return "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k.pth"

        case "span":
            return "2xHFA2kSPAN_27k.pth"

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
        
        case "realesrgan":
            return "2xHFA2kShallowESRGAN.pth"
        
        case "nafnet":
            return "NAFNet-GoPro-width64.pth"
        
        case "span-denoise":
            return "1x_span_anime_pretrain.pth"
        
        case "gmfss":
            return "gmfss-fortuna-union.zip"
        
        case _:
            raise ValueError(f"Model {model} not found.")

def downloadAndLog(model: str, filename: str, download_url: str, folderPath: str):
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
    model: str = None, cuganKind: str = None, upscaleFactor: int = 2
) -> str:
    os.makedirs(weightsDir, exist_ok=True)
    
    filename = modelsMap(model, upscaleFactor, cuganKind)

    match model:
        case "cugan" | "shufflecugan":
            cuganFolderPath = os.path.join(weightsDir, model)
            os.makedirs(cuganFolderPath, exist_ok=True)
            fullUrl = f"{url if model == 'shufflecugan' else cuganUrl}{filename}"
            return downloadAndLog(model, filename, fullUrl, cuganFolderPath)

        case "compact" | "ultracompact" | "superultracompact":
            compactFolderPath = os.path.join(weightsDir, model)
            os.makedirs(compactFolderPath, exist_ok=True)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, compactFolderPath)

        case "span":
            spanFolderPath = os.path.join(weightsDir, "span")
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

        case "rife" | "rife4.14" | "rife4.14-lite" | "rife4.13-lite" | "rife4.6":
            # The model names inside the repo are different from the ones used in the code
            model_mapping = {
                "rife": "rife414",
                "rife4.14": "rife414",
                "rife4.14-lite": "rife414lite",
                "rife4.13-lite": "rife413lite",
                "rife4.6": "rife46",
            }
            model = model_mapping.get(model, model)
            rifeFolderPath = os.path.join(weightsDir, "rife", model)
            os.makedirs(rifeFolderPath, exist_ok=True)
            filename = f"rife{model[4:]}.pkl"
            fullUrl = f"{url}{filename}"
            outFileName = "flownet.pkl"
            return downloadAndLog(model, outFileName, fullUrl, rifeFolderPath)

        case "segment":
            segmentFolderPath = os.path.join(weightsDir, "segment")
            os.makedirs(segmentFolderPath, exist_ok=True)
            filename = "isnetis.ckpt"
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, segmentFolderPath)

        case "realesrgan":
            realesrganFolderPath = os.path.join(weightsDir, "realesrgan")
            os.makedirs(realesrganFolderPath, exist_ok=True)
            filename = "2xHFA2kShallowESRGAN.pth"
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

        case _:
            print(f"Model {model} not found.")
            return None
