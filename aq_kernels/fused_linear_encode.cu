// Fused linear AQ encode v8 - adaptive strategy:
//   Small N (decode): single-kernel fully-fused encode (no cuBLAS, 1 launch)
//   Large N (prefill): L2 cache tiling with cuBLAS + fused post-processing
//
// The small-N kernel does GEMM + argmax + codebook sub for ALL M stages
// in one kernel launch, with residual living in shared memory.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>

#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif

namespace {

constexpr int DEFAULT_TILE_N = 65536;
constexpr int SMALL_N_THRESHOLD = 1024;

// Fully fused encode kernel for small N (decode)
// One block per vector, K=256 threads. Residual stays in shared memory across
// all M stages. Each thread computes one logit (dot product), then warp-reduce
// for argmax, then cooperatively sub codebook row from shared residual.
__global__ void fully_fused_encode_kernel(
    const __half* __restrict__ x,           // [N, d]
    const __half* __restrict__ weight,      // [M, d, K]
    const __half* __restrict__ bias,        // [M, K]
    const __half* __restrict__ codebooks,   // [M, K, d]
    uint8_t* __restrict__ codes,            // [N, M]
    int N, int M, int K, int d)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int num_warps = blockDim.x >> 5;

    extern __shared__ char smem[];
    __half* s_res  = reinterpret_cast<__half*>(smem);
    float*  s_wval = reinterpret_cast<float*>(s_res + d);
    int*    s_widx = reinterpret_cast<int*>(s_wval + num_warps);

    for (int j = tid; j < d; j += K)
        s_res[j] = x[(int64_t)n * d + j];
    __syncthreads();

    for (int m = 0; m < M; ++m) {
        const __half* w_m = weight + (int64_t)m * d * K;
        float acc = __half2float(bias[(int64_t)m * K + tid]);
        for (int j = 0; j < d; j += 2) {
            float r0 = __half2float(s_res[j]);
            float r1 = __half2float(s_res[j + 1]);
            acc += r0 * __half2float(w_m[(int64_t)j * K + tid]);
            acc += r1 * __half2float(w_m[(int64_t)(j + 1) * K + tid]);
        }

        float my_val = acc;
        int   my_idx = tid;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float o_v = __shfl_xor_sync(0xffffffff, my_val, mask);
            int   o_i = __shfl_xor_sync(0xffffffff, my_idx, mask);
            if (o_v > my_val) { my_val = o_v; my_idx = o_i; }
        }
        if (lane == 0) {
            s_wval[warp_id] = my_val;
            s_widx[warp_id] = my_idx;
        }
        __syncthreads();

        int winner;
        if (tid == 0) {
            float bv = s_wval[0];
            int   bi = s_widx[0];
            for (int w = 1; w < num_warps; ++w)
                if (s_wval[w] > bv) { bv = s_wval[w]; bi = s_widx[w]; }
            s_widx[0] = bi;
            codes[(int64_t)n * M + m] = (uint8_t)bi;
        }
        __syncthreads();
        winner = s_widx[0];

        const __half* cb_row = codebooks + ((int64_t)m * K + winner) * d;
        for (int j = tid; j < d; j += K) {
            float r = __half2float(s_res[j]);
            r -= __half2float(cb_row[j]);
            s_res[j] = __float2half(r);
        }
        __syncthreads();
    }
}

// Post-processing kernel for large N (used with cuBLAS GEMM)
__global__ void fused_argmax_gather_sub_kernel(
    const __half* __restrict__ logits,
    const __half* __restrict__ codebook_m,
    __half* __restrict__ residual,
    uint8_t* __restrict__ codes_base,
    int tile_N, int K, int d, int M_stride, int m_col)
{
    const int n = blockIdx.x;
    if (n >= tile_N) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int num_warps = blockDim.x >> 5;

    extern __shared__ char smem[];
    float* s_warp_val = (float*)smem;
    int*   s_warp_idx = (int*)(s_warp_val + num_warps);

    float my_val = (tid < K) ? __half2float(logits[(int64_t)n * K + tid]) : -1e30f;
    int   my_idx = tid;

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float o_v = __shfl_xor_sync(0xffffffff, my_val, mask);
        int   o_i = __shfl_xor_sync(0xffffffff, my_idx, mask);
        if (o_v > my_val) { my_val = o_v; my_idx = o_i; }
    }
    if (lane == 0) {
        s_warp_val[warp_id] = my_val;
        s_warp_idx[warp_id] = my_idx;
    }
    __syncthreads();

    int winner;
    if (tid == 0) {
        float bv = s_warp_val[0];
        int   bi = s_warp_idx[0];
        for (int w = 1; w < num_warps; ++w) {
            if (s_warp_val[w] > bv) { bv = s_warp_val[w]; bi = s_warp_idx[w]; }
        }
        s_warp_idx[0] = bi;
        codes_base[n * M_stride + m_col] = (uint8_t)bi;
    }
    __syncthreads();
    winner = s_warp_idx[0];

    const __half* cb_row = codebook_m + (int64_t)winner * d;
    __half* res_row = residual + (int64_t)n * d;
    const int d2 = d >> 1;
    const __half2* cb2 = reinterpret_cast<const __half2*>(cb_row);
    __half2* res2 = reinterpret_cast<__half2*>(res_row);
    for (int j = tid; j < d2; j += blockDim.x) {
        res2[j] = __hsub2(res2[j], cb2[j]);
    }
}

void check_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
}

}  // namespace

// Adaptive forward: small N -> single fused kernel, large N -> L2 tiled
torch::Tensor fused_linear_encode_forward(
    torch::Tensor x, torch::Tensor weight_t,
    torch::Tensor bias, torch::Tensor codebooks,
    bool /*fp32_residual*/, int user_tile_n)
{
    check_cuda(x, "x");
    const int N = (int)x.size(0);
    const int d = (int)x.size(1);
    const int M = (int)weight_t.size(0);
    const int K = (int)weight_t.size(2);

    auto to_h = [](const torch::Tensor& t) {
        if (t.scalar_type() == torch::kFloat16 && t.is_contiguous()) return t;
        return t.to(torch::kFloat16).contiguous();
    };
    auto wt = to_h(weight_t);
    auto bi = to_h(bias);
    auto cb = to_h(codebooks);

    auto codes = torch::empty({N, M}, x.options().dtype(torch::kUInt8));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream(x.device().index());

    if (N <= SMALL_N_THRESHOLD) {
        // Fully fused single-kernel path
        auto x_h = to_h(x);
        const int threads = K;
        const int num_warps = threads >> 5;
        const int smem = d * (int)sizeof(__half) + num_warps * ((int)sizeof(float) + (int)sizeof(int));

        fully_fused_encode_kernel<<<N, threads, smem, stream>>>(
            reinterpret_cast<const __half*>(x_h.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(wt.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bi.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(cb.data_ptr<at::Half>()),
            codes.data_ptr<uint8_t>(),
            N, M, K, d);
        return codes;
    }

    // L2 tiled path (prefill)
    auto residual = to_h(x).clone();
    const int tile_N = std::min(N, (user_tile_n > 0) ? user_tile_n : DEFAULT_TILE_N);
    auto logits = torch::empty({tile_N, K}, residual.options());

    const int threads = K;
    const int num_warps = threads >> 5;
    const int smem = num_warps * ((int)sizeof(float) + (int)sizeof(int));

    for (int t_start = 0; t_start < N; t_start += tile_N) {
        const int t_end = std::min(t_start + tile_N, N);
        const int t_N = t_end - t_start;
        auto res_tile = residual.narrow(0, t_start, t_N);
        auto logits_tile = (t_N == tile_N) ? logits : logits.narrow(0, 0, t_N);

        for (int m = 0; m < M; ++m) {
            torch::addmm_out(logits_tile, bi[m], res_tile, wt[m]);
            fused_argmax_gather_sub_kernel<<<t_N, threads, smem, stream>>>(
                reinterpret_cast<const __half*>(logits_tile.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(cb[m].data_ptr<at::Half>()),
                reinterpret_cast<__half*>(res_tile.data_ptr<at::Half>()),
                codes.data_ptr<uint8_t>() + (int64_t)t_start * M,
                t_N, K, d, M, m);
        }
    }
    return codes;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_encode_forward", &fused_linear_encode_forward,
          py::arg("x"), py::arg("weight_t"), py::arg("bias"),
          py::arg("codebooks"), py::arg("fp32_residual") = true,
          py::arg("tile_n") = 0);
}
