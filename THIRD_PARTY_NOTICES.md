# Third-Party Notices

This product uses third-party software. Notices required by their licenses
are reproduced below.

## Depth Anything 3 / DA3-Streaming

Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
Copyright 2025 The Depth Anything 3 Team.
The vendored DINOv2 backbone also retains its Meta Platforms copyright notices.

The existing DA3 runtime and the DA3 Small/Base model weights used by our
depth-video streaming integration are licensed under Apache License 2.0.
See [license](src/depth/depth_anything_3/LICENSE) and
[integration scope and provenance](docs/DA3_STREAMING.md).

The chunk scheduler and depth-scale alignment are a TAS implementation inspired
by DA3-Streaming. No SALAD, upstream loop-closure code, or non-commercial
Giant/Nested/Large any-view checkpoint is included in this integration.

## NVIDIA Video Effects SDK (Maxine VSR)

This software contains source code provided by NVIDIA Corporation.

Licensed under the NVIDIA Software License Agreement and AI Product-Specific
Terms. Bundled model weights are licensed under the NVIDIA Open Model
License Agreement.

- https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/
- https://developer.download.nvidia.com/licenses/NVIDIA-Open-Model-License.pdf

## Limbo V2 and Cyte V1 model weights

Models by NevermindNilas, copyright (c) 2026. Both are CC BY-NC 4.0
with an additional commercial-use grant for TheAnimeScripter users.
See the authoritative [Limbo V2 license](https://github.com/NevermindNilas/Ai-models/blob/main/Limbo-V2/LICENSE.MD)
and [Cyte license](https://github.com/NevermindNilas/Ai-models/blob/main/Cyte-V1-SuperUltraCompact/LICENSE.MD).
Limbo's upstream DA3 material retains its Apache-2.0 terms;
Cyte uses the BSD-3-Clause Real-ESRGAN SRVGGNetCompact architecture.
