// Softsplat forward kernel. Port of niklaus/softmax-splatting CuPy impl.
// One thread per source pixel; each thread scatters into 4 bilinear corners
// of the output via gpuAtomicAdd. Templated for fp32 / fp16 dispatch.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void softsplat_forward_kernel(
    const int n,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ flow,
    scalar_t* __restrict__ output,
    const int batch,
    const int channels,
    const int height,
    const int width)
{
    const int hw = height * width;
    const int chw = channels * hw;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x)
    {
        const int x = idx % width;
        const int y = (idx / width) % height;
        const int c = (idx / hw) % channels;
        const int b = idx / chw;

        const int flow_x_off = ((b * 2 + 0) * height + y) * width + x;
        const int flow_y_off = ((b * 2 + 1) * height + y) * width + x;

        const float fx = static_cast<float>(x) + static_cast<float>(flow[flow_x_off]);
        const float fy = static_cast<float>(y) + static_cast<float>(flow[flow_y_off]);

        if (!isfinite(fx) || !isfinite(fy)) continue;

        const int nw_x = static_cast<int>(floorf(fx));
        const int nw_y = static_cast<int>(floorf(fy));
        const int ne_x = nw_x + 1;
        const int ne_y = nw_y;
        const int sw_x = nw_x;
        const int sw_y = nw_y + 1;
        const int se_x = nw_x + 1;
        const int se_y = nw_y + 1;

        const float w_nw = (static_cast<float>(se_x) - fx) * (static_cast<float>(se_y) - fy);
        const float w_ne = (fx - static_cast<float>(sw_x)) * (static_cast<float>(sw_y) - fy);
        const float w_sw = (static_cast<float>(ne_x) - fx) * (fy - static_cast<float>(ne_y));
        const float w_se = (fx - static_cast<float>(nw_x)) * (fy - static_cast<float>(nw_y));

        const float in_val = static_cast<float>(input[idx]);
        const int bc_off = (b * channels + c) * hw;

        if (nw_x >= 0 && nw_x < width && nw_y >= 0 && nw_y < height) {
            const int out_off = bc_off + nw_y * width + nw_x;
            gpuAtomicAdd(&output[out_off], static_cast<scalar_t>(in_val * w_nw));
        }
        if (ne_x >= 0 && ne_x < width && ne_y >= 0 && ne_y < height) {
            const int out_off = bc_off + ne_y * width + ne_x;
            gpuAtomicAdd(&output[out_off], static_cast<scalar_t>(in_val * w_ne));
        }
        if (sw_x >= 0 && sw_x < width && sw_y >= 0 && sw_y < height) {
            const int out_off = bc_off + sw_y * width + sw_x;
            gpuAtomicAdd(&output[out_off], static_cast<scalar_t>(in_val * w_sw));
        }
        if (se_x >= 0 && se_x < width && se_y >= 0 && se_y < height) {
            const int out_off = bc_off + se_y * width + se_x;
            gpuAtomicAdd(&output[out_off], static_cast<scalar_t>(in_val * w_se));
        }
    }
}

at::Tensor softsplat_forward(const at::Tensor& input, const at::Tensor& flow)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(flow.is_cuda(), "flow must be CUDA");
    TORCH_CHECK(input.dim() == 4, "input must be 4D [N,C,H,W]");
    TORCH_CHECK(flow.dim() == 4 && flow.size(1) == 2, "flow must be [N,2,H,W]");
    TORCH_CHECK(input.size(0) == flow.size(0) &&
                input.size(2) == flow.size(2) &&
                input.size(3) == flow.size(3),
                "input and flow must match in N,H,W");
    TORCH_CHECK(input.scalar_type() == flow.scalar_type(),
                "input and flow must have the same dtype");

    const c10::cuda::CUDAGuard device_guard(input.device());

    auto input_c = input.contiguous();
    auto flow_c = flow.contiguous();

    const int batch = input_c.size(0);
    const int channels = input_c.size(1);
    const int height = input_c.size(2);
    const int width = input_c.size(3);
    const int n = batch * channels * height * width;

    auto output = at::zeros_like(input_c);

    const int threads = 512;
    const int blocks_max = 65535;
    const int blocks = std::min(blocks_max, (n + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_c.scalar_type(), "softsplat_forward", [&] {
        softsplat_forward_kernel<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()
        >>>(
            n,
            input_c.data_ptr<scalar_t>(),
            flow_c.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch, channels, height, width
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softsplat_forward, "softsplat forward (CUDA)");
}
