*******************************************************
*                    The Anime Scripter                *
*******************************************************

Creator: NevermindNilas
Discord: https://discord.gg/GkCrawMuZ6
Github: https://github.com/NevermindNilas/TheAnimeScripter

The Anime Scripter is provided "as is", without any warranty. For any issues, please report them to Nilas via the Discord server. For more information about the license, refer to License.md.

# System Requirements:

## Minimum:
    - OS: Windows 10/11
    - CPU: Quad-Core CPU
    - RAM: 16 GB
    - GPU: Any GPU with DirectX 12 and Vulkan support
    - Disk Space: Approximately 8 GB

## Recommended:
    - OS: Windows 10/11
    - CPU: Octa-Core CPU
    - RAM: 32 GB
    - GPU: NVidia 3000 Series or higher / AMD 6000 Series or higher
    - Disk Space: Approximately 8 GB

## Recommended GPUs:
    - For NCNN, virtually any VULKAN capable GPUs.
    - For DirectML, any AMD RX500+ series, any Intel GPU and any NVIDIA 900+ series.
    - For CUDA, Any NVIDIA 1000+ series.
    - For TensorRT, Any NVIDIA RTX 2000 series GPU

# Installation:
    Please refer to the installation tutorial or build the .exe file yourself using the guide available at: 
    https://github.com/NevermindNilas/TheAnimeScripter?tab=readme-ov-file#%EF%B8%8F-getting-started

# Important Notes:
    - TensorRT Engines may take a couple of minutes to build, but they are reused once built. The performance benefits are well worth the wait.
    - AMD / Intel GPU users are advised to use the NCNN or DirectML Versions of upscalers / interpolators for optimal performance and to avoid falling back to CPU.
    - GMFSS may experience VRAM limitations for high-resolution clips; ensure your GPU has 8 GB or more VRAM for 1080p footage.
    - Enabling the Rife Ensemble triggers internal calls to the warp algorithm, providing better results at the cost of some performance. Enable it if the performance loss is acceptable.
    - The resize multiplier supports fractional values like 0.5 for downscaling the clip (3840x2160 -> 1920x1080).
    - All upscalers, have 2x models (1920x1080 -> 3840x2160).
    - GMFSS, Depth Map, and Segmentation require CUDA-capable GPUs for acceleration; otherwise, the script will fallback to CPU with potential performance issues.
    - The Denoising algorithms are extremely slow by nature and require high-end GPUs for good performance. NVidia 3090/4080 is recommended.