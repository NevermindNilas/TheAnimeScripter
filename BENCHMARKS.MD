### Benchmarks ran with the latest .exe file at the time, benchmark.py or benchmark.exe script, no encoding and on a 1920x1080p video with FP16 or BF16 on for all runs.

#### Note, since these are ran in order with no time gaps inbetween there's a possibility for GPU heat becoming a bottleneck so these results may not be fully representative of the real world performance.

#### If the result is 0.0 / Null that's typically because of Out Of VRAM Issues ( like GMFSS ) or the GPU is not supported

V3 -> V4, Removed Dedup and Denoise, made the benchmark more dynamic based on OS and GPU available, NVIDIA will only test CUDA and TensorRT whilst AMD will only test NCNN and DirectML
V2 -> V3, Added new options and a different video with more duplicated frames for a more accurate Dedup result, modified certain benchmarks options like GMFSS and OmniSR To be shorter due to the innate slow inference speed.
V1 -> V2, Longer video for more accurate results, More time in between runs to cool of the gpu, more options ( dedup )

#### Testing Methodology V4
# Benchmark Results

## System Information

- **CPU:** 13th Gen Intel(R) Core(TM) i7-13700K
- **Total RAM:** 31.77 GB
- **Graphics Card 0:** Intel(R) UHD Graphics 770
- **Graphics Card 1:** NVIDIA GeForce RTX 3090

**Version:** 2.0.0 ( Pre-Release )
**Testing Methodology:** V4.4

## Upscale Results

| Model                      | Time (s) | FPS   |
|----------------------------|----------|-------|
| shufflecugan               | 30.65    | 23.49 |
| compact                    | 50.81    | 14.17 |
| ultracompact               | 29.01    | 24.82 |
| superultracompact          | 9.96     | 72.31 |
| span                       | 57.32    | 12.56 |
| compact-tensorrt           | 44.26    | 21.69 |
| ultracompact-tensorrt      | 24.63    | 38.97 |
| superultracompact-tensorrt | 9.80     | 97.99 |
| shufflecugan-tensorrt      | 23.42    | 40.98 |
| span-tensorrt              | 36.82    | 26.08 |

## Interpolate Results

| Model                           | Time (s) | FPS    |
|---------------------------------|----------|--------|
| rife4.6                         | 16.46    | 145.78 |
| rife4.6-ensemble                | 22.20    | 86.50  |
| rife4.22                        | 23.34    | 102.82 |
| rife4.22-ensemble               | 18.85    | 101.83 |
| rife4.22-lite                   | 17.16    | 139.84 |
| rife4.22-lite-ensemble          | 14.44    | 132.97 |
| rife4.6-tensorrt                | 8.83     | 271.90 |
| rife4.6-tensorrt-ensemble       | 12.38    | 155.06 |
| rife4.22-tensorrt               | 15.50    | 154.86 |
| rife4.22-tensorrt-ensemble      | 12.72    | 150.94 |
| rife4.22-lite-tensorrt          | 11.55    | 207.73 |
| rife4.22-lite-tensorrt-ensemble | 9.54     | 201.18 |

# Toji
![benchmarkResults](https://github.com/user-attachments/assets/195e38da-b3d8-45af-868c-def2b3034e7f)

# Sergio
![benchmarkResults](https://github.com/user-attachments/assets/de2a0573-1c0a-46a3-a4e0-f02fa5f13e85)

# Nilas
![benchmarkResults](https://github.com/user-attachments/assets/c43b23fc-e8fc-4f1b-8749-d88130617a36)

# 13700K OC + UV, 3090 OC, 32GB DDR5 6400MHz CL 30, TAS 1.8.0 Windows 11, Nilas
![Untitled](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/4b99b774-e3f1-482a-a4cc-4b07da80521d)

#### Testing Methodology V3
# 13900k, 4090, 32GB DDR5 4800MHz, TAS 1.7.2, Atlas OS ( Modified W11 ), Sergio.
![nilasedits (10)](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/83681671-06ec-41b6-bb90-52d33c315890)

# 7950x, 4090, 64GB DDR5, 1.7.2, Linux, Sudo
![nilasedits (8)](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/36cf3510-02f2-45fb-8821-bf4219e9a133)

# 7950x, 4090, 64GB DDR5, Pre-Release 1.7.1, Linux, Sudo
![nilasedits (7)](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/8adf6915-9d38-4ab8-8492-c8d74107b118)


# 13900k, 4090, 32GB DDR5 4800MHz, TAS 1.7.0, KernelOS ( Modified W11 ), Sergio.
![nilasedits (6)](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/b91efa6f-e9a8-4a80-b523-76886fcb7f25)


# 13700K Undervolted + OC, 3090 OC, 32GB DDR5 6400mhz CL 30, TAS 1.7.0, Nilas
![nilasedits (5)](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/053915c5-9530-4629-a925-4e685311b11c)

-------------------------------------------------------------------------------------------------------------
# Ryzen 9 7950X, RTX 4090, 64GB DDR5, *Pre-Release* TAS 1.6.4, Sudo
![nilasedits (4)](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/0be8254e-94b8-4106-a1ac-5a44d68bf4fa)

# 13900k, 4090, 32GB DDR5 4800MHz, TAS 1.6.3, Sergio.
![nilasedits (2)](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/2b3901f8-662f-4e83-8704-f49e1ab42549)

# Ryzen 9 7950X3D, RTX 4090, 64GB DDR5, TAS 1.6.3, Toji.
![nilasedits (3)](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/302976cd-7b2b-4cfa-a7d4-41aa3384014b)


# 13700k Undervolted + Overclocked, 3090 Overclocked, 32GB DDR5 7200Mhz CL 38, *PRERELEASE* TAS 1.6.3, Nilas
![nilasedits](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/67ae29a1-deac-4ced-964c-47a50400cd8e)

-------------------------------------------------------------------------------------------------------------
#### Testing Methodology V2
-------------------------------------------------------------------------------------------------------------

# 13700k Undervolted + Overclocked, 3090 Overclocked, 32GB DDR5 6800Mhz, TAS 1.6.0, Nilas.

![tables](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/a00d7d09-e6d1-4f94-8824-068a9f2f2213)

# AMD 3700x, RTX 3080, 24GB DDR4 3600MHz, TAS 1.6.0, Windows VM ( 7C/14Threads ), TNTWise.

![tables](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/b7ca8ced-b034-4ef1-8218-2ed5cd0842c1)


-------------------------------------------------------------------------------------------------------------
#### Testing Methodology V1
-------------------------------------------------------------------------------------------------------------

# 13700k Undervolted + Overclocked, 3090 Overclocked, 32GB DDR5 6800Mhz, TAS 1.5.0, Nilas.

![tables](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/887177e2-f356-4640-b5d7-abf2e3056071)

# AMD 3700x, RTX 3080, 24GB DDR4 3600MHz, TAS 1.5.0, Windows VM ( 7C/14Threads ), TNTWise.

![tables](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/71a326d5-6f6a-404a-93b6-bdf19a56b385)


# AMD 7800x3D, RTX 4080, 32GB DDR5, TAS 1.5.0, Kovel

![tables](https://cdn.discordapp.com/attachments/1208539289328025621/1224397781205127359/image.png?ex=661d584b&is=660ae34b&hm=fe8baf72d44b733873073729e22c50efdf63a094c3d1f1da98220e8583eb18c0&)


# AMD 5800X, RTX 3080, 32GB DDR4, TAS 1.5.0, HUK

![nilasedits](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/7a0cfd32-3083-4f15-a88d-3876962106a4)
