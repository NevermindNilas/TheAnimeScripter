import wget
import os
import logging

dirPath = os.path.dirname(__file__)
weightsDir = os.path.join(dirPath, "weights")
url = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/"
cuganUrl = (
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"
)


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

    match model:
        case "cugan" | "shufflecugan":
            cuganFolderPath = os.path.join(weightsDir, "cugan")
            os.makedirs(cuganFolderPath, exist_ok=True)
            filename = (
                "sudo_shuffle_cugan_9.584.969.pth"
                if model == "shufflecugan"
                else f"cugan_up{upscaleFactor}x-latest-{cuganKind}.pth"
            )
            fullUrl = f"{url if model == 'shufflecugan' else cuganUrl}{filename}"
            return downloadAndLog(model, filename, fullUrl, cuganFolderPath)

        case "compact" | "ultracompact" | "superultracompact":
            compactFolderPath = os.path.join(weightsDir, "compact")
            os.makedirs(compactFolderPath, exist_ok=True)
            filenameMap = {
                "compact": "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth",
                "ultracompact": "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k.pth",
                "superultracompact": "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k.pth",
            }
            filename = filenameMap.get(model)
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, compactFolderPath)

        case "span":
            spanFolderPath = os.path.join(weightsDir, "span")
            os.makedirs(spanFolderPath, exist_ok=True)
            filename = "2xHFA2kSPAN_27k.pth"
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, spanFolderPath)

        case "swinir":
            swinirFolderPath = os.path.join(weightsDir, "swinir")
            os.makedirs(swinirFolderPath, exist_ok=True)
            filename = "2xHFA2kSwinIR-S.pth"
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, swinirFolderPath)

        case "omnisr":
            omnisrFolderPath = os.path.join(weightsDir, "omnisr")
            os.makedirs(omnisrFolderPath, exist_ok=True)
            filename = "2xHFA2kOmniSR.pth"
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, omnisrFolderPath)

        case "gmfss":
            gmfssFolderPath = os.path.join(weightsDir, "gmfss")
            os.makedirs(gmfssFolderPath, exist_ok=True)
            filename = "gmfss-fortuna-union.zip"
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
            filename = "scunet_color_real_psnr.pth"
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, scunetFolderPath)
        
        case "nafnet":
            nafnetFolderPath = os.path.join(weightsDir, "nafnet")
            os.makedirs(nafnetFolderPath, exist_ok=True)
            filename = "NAFNet-GoPro-width64.pth"
            fullUrl = f"{url}{filename}"
            return downloadAndLog(model, filename, fullUrl, nafnetFolderPath)
        
        case _:
            print(f"Model {model} not found.")
            return None
