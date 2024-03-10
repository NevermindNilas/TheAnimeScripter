*****************************************************
*                The Anime Scripter                 *
*****************************************************

Creator: NevermindNilas
Discord: https://discord.gg/GkCrawMuZ6
Github: https://github.com/NevermindNilas/TheAnimeScripter

!The Anime Scripter is provided as is, without any warranty. Feel free to join the Discord server and report any issues to Nilas. Learn more about the license in License.md.!

# Minimum System Requirements:
    - CPU: Quad Core CPU
    - RAM: 16 GB
    - GPU: Vulkan Capable GPU with 4GB VRAM
    - Disk Space: Approximately 6 GB

# Recommended System Requirements:
    - CPU: Octa Core CPU
    - RAM: 32 GB
    - GPU: NVidia 1000 series / RX 6000 series
    - Disk Space: Approximately 6 GB

# Installation:
    Refer to the installation tutorial or self-build the .exe file using the guide available at: 
    https://github.com/NevermindNilas/TheAnimeScripter?tab=readme-ov-file#%EF%B8%8F-getting-started

# Things to consider:
    - The Script will Utilize CUDA acceleration automatically if you have an NVIDIA GPU of series 1000 or higher for optimal performance.
    - AMD / Intel GPU users should use the NCNN Versions of upscalers / interpolators to avoid falling back to CPU and ensure optimal performance.
    - GMFSS may experience VRAM limitations for high-resolution clips; ensure your GPU has 8 GB or more VRAM for 1080p footage.
    - Enabling the Rife Ensemble triggers internal calls to the warp algorithm, providing better results with some loss in performance. Enable it if the performance loss is acceptable.
    - The resize multiplier supports fractional values like 0.5 for downscaling the clip (3840x2160 -> 1920x1080).
    - All upscalers, except CUGAN, have 2x models (1920x1080 -> 3840x2160). CUGAN can go up to 4x but expect significant slowdowns depending on the clip's resolution.
    - GMFSS, Depth Map, and Segmentation require CUDA-capable GPUs for acceleration; otherwise, the script will fallback to CPU with potential performance issues.
    - The Denoising algorithms are extremely slow by nature and require high end GPUs for good performances, recommended NVidia 3090/4080.
