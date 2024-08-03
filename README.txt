*******************************************************
*                    The Anime Scripter                *
*******************************************************

Creator: 
    NevermindNilas

Discord: 
    Join our community on Discord - https://discord.gg/GkCrawMuZ6

GitHub:
    Explore the project - https://github.com/NevermindNilas/TheAnimeScripter

Disclaimer: 
    The Anime Scripter is provided "as is", without warranty of any kind. For issues or suggestions, please report them via our Discord server. For license details, refer to License.md.

Overview:
    The Anime Scripter (TAS) is a cutting-edge tool designed for After Effects 2024, ensuring the best compatibility and performance. 
    While TAS strives for backward compatibility, optimal performance is not guaranteed on older versions.

System Requirements:

Minimum:
    - OS: Windows 10/11
    - CPU: Quad-Core
    - RAM: 16 GB
    - GPU: Supports DirectX 12 and Vulkan
    - Disk Space: ~12 GB

Recommended:
    - OS: Windows 10/11, fully updated
    - CPU: Octa-Core
    - RAM: 32 GB
    - GPU: NVidia 3000 Series / AMD 6000 Series or better
    - Disk Space: ~12 GB

Recommended GPUs:
    - NCNN / DirectML: Compatible with all Vulkan / DirectX 12 capable GPUs.
    - CUDA: Requires NVIDIA 1000+ series.
    - TensorRT: Requires NVIDIA RTX 2000+ series GPUs.

Installation:
    Get started by following our detailed installation guide. Visit our GitHub page for step-by-step instructions: 
    https://github.com/NevermindNilas/TheAnimeScripter?tab=readme-ov-file#%EF%B8%8F-getting-started

Important Notes:
    - Building TensorRT Engines is a one-time process that may take a few minutes but significantly boosts performance.
    - For AMD / Intel GPU users: Opt for NCNN or DirectML versions to achieve the best performance.
    - The Rife Ensemble feature enhances results at a slight performance cost. Enable it if the trade-off is acceptable.
    - Resize multiplier now supports fractional values (e.g., 0.5 for downscaling from 3840x2160 to 1920x1080).
    - Upscalers include 2x models for converting 1920x1080 to 3840x2160.
    - Features like GMFSS, Depth Map, and Segmentation acceleration require CUDA / TensorRT -capable GPUs. Without these, processing defaults to CPU, impacting performance significantly.
    - Denoising is resource-intensive and performs best on high-end GPUs, such as NVidia 3090/4080.