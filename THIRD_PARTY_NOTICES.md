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

## Limbo V2, Limbo V3 and Cyte V1 model weights

Models by NevermindNilas, copyright (c) 2026. All are CC BY-NC 4.0
with an additional commercial-use grant for TheAnimeScripter users.
See the authoritative [Limbo V2 license](https://github.com/NevermindNilas/Ai-models/blob/main/Limbo-V2/LICENSE.MD),
[Limbo V3 license](https://github.com/NevermindNilas/Ai-models/blob/main/Limbo-V3/LICENSE.MD)
and [Cyte license](https://github.com/NevermindNilas/Ai-models/blob/main/Cyte-V1-SuperUltraCompact/LICENSE.MD).
Limbo's upstream DA3 material retains its Apache-2.0 terms;
Cyte uses the BSD-3-Clause Real-ESRGAN SRVGGNetCompact architecture.

## SMoVSR (2x_umzi_anime_smovsr)

Copyright (c) 2026 enhancr. Architecture and weights by umzi, MIT License.
`src/extraArches/smovsr.py` is adapted from
[enhancr/SMoVSR](https://github.com/enhancr/SMoVSR) `smovsr_arch.py`; the
streaming window and the ONNX exports TAS downloads are TAS additions.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
