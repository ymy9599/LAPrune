#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>

// PyTorch's extension build may pass these macros which disable CUDA half/half2 operators.
// We explicitly enable half/half2 intrinsics for our fp16 kernel.
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace {

template <typename code_t>
__device__ __forceinline__ int aq_read_code_as_index(const code_t v) {
    // Treat code as an unsigned 8-bit index when stored in 8-bit tensors.
    // - For uint8: value is already 0..255
    // - For int8 : reinterpret bits as unsigned, so e.g. -1 -> 255
    // - For int64: low 8 bits are used (safe when K<=256 and upstream stores valid indices)
    return static_cast<int>(static_cast<unsigned char>(v));
}

template <typename scalar_t, typename code_t>
__global__ void aq_lut_kernel(
    const scalar_t* __restrict__ q,          // [Q, d]
    const scalar_t* __restrict__ codebooks,  // [M, K, d]
    const code_t* __restrict__ codes,        // [Q, N, M] (int64/uint8/int8)
    scalar_t* __restrict__ scores,           // [Q, N]
    int Q,
    int N,
    int M,
    int K,
    int d
) {
    int q_idx = blockIdx.x;
    int n_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (q_idx >= Q || n_idx >= N) {
        return;
    }

    // Cache q_row in shared memory once per block (all threads share same q_idx).
    extern __shared__ unsigned char smem[];
    scalar_t* q_shared = reinterpret_cast<scalar_t*>(smem);
    const scalar_t* q_row = q + static_cast<long long>(q_idx) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        q_shared[i] = q_row[i];
    }
    __syncthreads();

    scalar_t acc = static_cast<scalar_t>(0);

    // 对每个子码本 m 累加 q·C_m[z_{q,n,m}]
    for (int m = 0; m < M; ++m) {
        long long code_offset = (static_cast<long long>(q_idx) * N + n_idx) * M + m;
        int k_idx = aq_read_code_as_index<code_t>(codes[code_offset]);
        if (k_idx < 0 || k_idx >= K) {
            continue;
        }

        long long cb_offset = (static_cast<long long>(m) * K + static_cast<int>(k_idx)) * d;
        const scalar_t* cb_row = codebooks + cb_offset;

        for (int i = 0; i < d; ++i) {
            acc += q_shared[i] * cb_row[i];
        }
    }

    scores[static_cast<long long>(q_idx) * N + n_idx] = acc;
}

// fp16 specialization: fp16 inputs, fp32 accumulate, fp16 output
template <typename code_t>
__global__ void aq_lut_kernel_half(
    const half* __restrict__ q,          // [Q, d]
    const half* __restrict__ codebooks,  // [M, K, d]
    const code_t* __restrict__ codes,    // [Q, N, M] (int64/uint8/int8)
    half* __restrict__ scores,           // [Q, N]
    int Q,
    int N,
    int M,
    int K,
    int d
) {
    int q_idx = blockIdx.x;
    int n_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (q_idx >= Q || n_idx >= N) {
        return;
    }

    // Cache q_row in shared memory once per block to avoid re-reading q for each codebook m.
    extern __shared__ unsigned char smem[];
    half* q_shared = reinterpret_cast<half*>(smem);
    const half* q_row = q + static_cast<long long>(q_idx) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        q_shared[i] = q_row[i];
    }
    __syncthreads();
    float acc = 0.0f;

    // For each codebook m, accumulate dot(q, C_m[z])
    for (int m = 0; m < M; ++m) {
        long long code_offset = (static_cast<long long>(q_idx) * N + n_idx) * M + m;
        int k_idx = aq_read_code_as_index<code_t>(codes[code_offset]);
        if (k_idx < 0 || k_idx >= K) {
            continue;
        }

        long long cb_offset = (static_cast<long long>(m) * K + static_cast<int>(k_idx)) * d;
        const half* cb_row = codebooks + cb_offset;

        // Try half2 vectorization when aligned
        bool aligned = ((reinterpret_cast<uintptr_t>(q_shared) & 0x3) == 0) && ((reinterpret_cast<uintptr_t>(cb_row) & 0x3) == 0);
        int i = 0;
        if (aligned) {
            const half2* q2 = reinterpret_cast<const half2*>(q_shared);
            const half2* cb2 = reinterpret_cast<const half2*>(cb_row);
            int d2 = d / 2;
            #pragma unroll
            for (int j = 0; j < d2; ++j) {
                half2 prod = __hmul2(q2[j], cb2[j]);
                float2 f = __half22float2(prod);
                acc += f.x + f.y;
            }
            i = d2 * 2;
        }
        // Tail (or unaligned fallback)
        for (; i < d; ++i) {
            acc += __half2float(q_shared[i]) * __half2float(cb_row[i]);
        }
    }

    scores[static_cast<long long>(q_idx) * N + n_idx] = __float2half_rn(acc);
}

}  // namespace

// ---------------------------------------------------------------------------
// Sparse/pruned attention kernel for decode-only (Q=1).
//
// Computes (per (B,H)):
//   out[d] = sum_i softmax_i * V[idx[i], d]
// where softmax_i = exp((q·K[idx[i]])*scale - max) / sum_j exp(...)
//
// Notes:
// - Forward-only, inference use.
// - Optimized for small K_keep (e.g., ~100-300) and head_dim around 128.
// - Uses two-pass dot product (max then denom/out) to avoid storing scores.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float half_to_float(const half h) { return __half2float(h); }
__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 h) { return __bfloat162float(h); }

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(const float x);

template <>
__device__ __forceinline__ half float_to_scalar<half>(const float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(const float x) {
    return __float2bfloat16_rn(x);
}

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(const scalar_t v);

template <>
__device__ __forceinline__ float scalar_to_float<half>(const half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename scalar_t, typename index_t>
__global__ void sparse_attn_q1_kernel(
    const scalar_t* __restrict__ q,     // [BH, D]
    const scalar_t* __restrict__ k,     // [BH, T, D]
    const scalar_t* __restrict__ v,     // [BH, T, D]
    const index_t* __restrict__ idx,    // [BH, Kkeep]
    scalar_t* __restrict__ out,         // [BH, D]
    int BH,
    int T,
    int D,
    int Kkeep,
    float scale
) {
    const int bh = static_cast<int>(blockIdx.x);
    if (bh >= BH) return;

    // Shared memory layout:
    // - weights : float[Kkeep]   (scaled scores, then exp weights)
    // - idx_s   : int[Kkeep]
    // - scalars : float[2]       (max, denom)
    // - warp_buf: float[WARPS]   (block reductions)
    // - qf      : float[D]       (cached q vector as float, to avoid repeated global loads)
    extern __shared__ unsigned char smem[];
    float* weights = reinterpret_cast<float*>(smem);
    int* idx_s = reinterpret_cast<int*>(weights + Kkeep);
    float* scalars = reinterpret_cast<float*>(idx_s + Kkeep);
    float* warp_buf = scalars + 2;
    float* qf = warp_buf + (blockDim.x >> 5);

    const scalar_t* q_row = q + static_cast<long long>(bh) * D;
    const scalar_t* k_row = k + static_cast<long long>(bh) * T * D;
    const scalar_t* v_row = v + static_cast<long long>(bh) * T * D;
    const index_t* idx_row = idx + static_cast<long long>(bh) * Kkeep;

    // Load indices to shared (once).
    for (int i = threadIdx.x; i < Kkeep; i += blockDim.x) {
        long long t = static_cast<long long>(idx_row[i]);
        if (t < 0) t = 0;
        if (t >= T) t = T - 1;
        idx_s[i] = static_cast<int>(t);
    }
    __syncthreads();

    // Cache q as float in shared (once per block).
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        qf[d] = scalar_to_float<scalar_t>(q_row[d]);
    }
    __syncthreads();

    // ---------------------------------------
    // Pass 1: compute scaled scores in parallel
    // - partition keys across warps
    // - reduce dot product within warp over D
    // ---------------------------------------
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    for (int i = warp; i < Kkeep; i += warps) {
        const int t = idx_s[i];
        const scalar_t* k_vec = k_row + static_cast<long long>(t) * D;
        float partial = 0.0f;
        for (int d = lane; d < D; d += 32) {
            partial += qf[d] * scalar_to_float<scalar_t>(k_vec[d]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }
        if (lane == 0) {
            weights[i] = partial * scale;
        }
    }
    __syncthreads();

    // Pass 1b: max over scores (block reduction)
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < Kkeep; i += blockDim.x) {
        local_max = fmaxf(local_max, weights[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane == 0) warp_buf[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < warps) ? warp_buf[lane] : -INFINITY;
        for (int offset = 16; offset > 0; offset >>= 1) {
            m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset));
        }
        if (lane == 0) scalars[0] = m;
    }
    __syncthreads();
    const float max_s = scalars[0];

    // Pass 2: exp weights and denom (block reduction)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < Kkeep; i += blockDim.x) {
        float w = expf(weights[i] - max_s);
        weights[i] = w;
        local_sum += w;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) warp_buf[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float denom = (lane < warps) ? warp_buf[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            denom += __shfl_down_sync(0xffffffff, denom, offset);
        }
        if (lane == 0) {
            if (denom <= 0.0f) denom = 1.0f;
            scalars[1] = denom;
        }
    }
    __syncthreads();
    const float denom = scalars[1];
    // Compute output vector in parallel over D.
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < Kkeep; ++i) {
            const int t = idx_s[i];
            const scalar_t* v_vec = v_row + static_cast<long long>(t) * D;
            acc += weights[i] * scalar_to_float<scalar_t>(v_vec[d]);
        }
        acc /= denom;
        out[static_cast<long long>(bh) * D + d] = float_to_scalar<scalar_t>(acc);
    }
}

torch::Tensor sparse_attn_q1_forward(
    torch::Tensor q,    // [B, H, 1, D] or [B, H, D]
    torch::Tensor k,    // [B, H, T, D]
    torch::Tensor v,    // [B, H, T, D]
    torch::Tensor idx   // [B, H, Kkeep]
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && idx.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(q.get_device() == k.get_device() && q.get_device() == v.get_device() && q.get_device() == idx.get_device(), "all inputs must be on the same CUDA device");

    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be [B,H,T,D]");
    TORCH_CHECK(idx.dim() == 3, "idx must be [B,H,Kkeep]");
    TORCH_CHECK(q.dim() == 3 || q.dim() == 4, "q must be [B,H,D] or [B,H,1,D]");

    if (q.dim() == 4) {
        TORCH_CHECK(q.size(2) == 1, "q must have Q=1 when dim==4");
        q = q.squeeze(2);
    }

    const int64_t B = k.size(0);
    const int64_t H = k.size(1);
    const int64_t T = k.size(2);
    const int64_t D = k.size(3);
    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T && v.size(3) == D, "k and v must have the same shape");
    TORCH_CHECK(q.size(0) == B && q.size(1) == H && q.size(2) == D, "q must match [B,H,D]");
    TORCH_CHECK(idx.size(0) == B && idx.size(1) == H, "idx must match [B,H,Kkeep]");
    const int64_t Kkeep = idx.size(2);
    TORCH_CHECK(Kkeep > 0, "idx Kkeep must be > 0");
    TORCH_CHECK(Kkeep <= 2048, "idx Kkeep too large for this kernel");

    TORCH_CHECK(
        q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
        "q/k/v must have same dtype"
    );
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
        "sparse_attn_q1_forward currently supports fp16/bf16"
    );
    TORCH_CHECK(idx.scalar_type() == torch::kLong || idx.scalar_type() == torch::kInt, "idx must be int64 or int32");

    const c10::cuda::CUDAGuard device_guard(q.device());
    if (!q.is_contiguous()) q = q.contiguous();
    if (!k.is_contiguous()) k = k.contiguous();
    if (!v.is_contiguous()) v = v.contiguous();
    if (!idx.is_contiguous()) idx = idx.contiguous();

    // Flatten to [BH, ...] for kernel simplicity.
    const int BH = static_cast<int>(B * H);
    auto q2 = q.view({BH, D});
    auto k2 = k.view({BH, T, D});
    auto v2 = v.view({BH, T, D});
    auto idx2 = idx.view({BH, Kkeep});

    torch::Tensor out = torch::empty({BH, D}, q.options());

    const float scale = 1.0f / sqrtf(static_cast<float>(D));
    const int threads = 128;
    dim3 grid(BH);
    dim3 block(threads);
    const int warps = threads / 32;
    // shared: weights(float)*K + idx(int)*K + scalars(float)*(2+warps)
    const size_t shmem =
        static_cast<size_t>(Kkeep) * sizeof(float)
        + static_cast<size_t>(Kkeep) * sizeof(int)
        + static_cast<size_t>(2 + warps) * sizeof(float)
        + static_cast<size_t>(D) * sizeof(float);

    if (q.scalar_type() == torch::kFloat16) {
        if (idx.scalar_type() == torch::kLong) {
            sparse_attn_q1_kernel<half, int64_t><<<grid, block, shmem>>>(
                reinterpret_cast<const half*>(q2.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(k2.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(v2.data_ptr<at::Half>()),
                idx2.data_ptr<int64_t>(),
                reinterpret_cast<half*>(out.data_ptr<at::Half>()),
                BH, static_cast<int>(T), static_cast<int>(D), static_cast<int>(Kkeep), scale
            );
        } else {
            sparse_attn_q1_kernel<half, int32_t><<<grid, block, shmem>>>(
                reinterpret_cast<const half*>(q2.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(k2.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(v2.data_ptr<at::Half>()),
                idx2.data_ptr<int32_t>(),
                reinterpret_cast<half*>(out.data_ptr<at::Half>()),
                BH, static_cast<int>(T), static_cast<int>(D), static_cast<int>(Kkeep), scale
            );
        }
    } else {
        // bfloat16
        if (idx.scalar_type() == torch::kLong) {
            sparse_attn_q1_kernel<__nv_bfloat16, int64_t><<<grid, block, shmem>>>(
                reinterpret_cast<const __nv_bfloat16*>(q2.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(k2.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(v2.data_ptr<at::BFloat16>()),
                idx2.data_ptr<int64_t>(),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                BH, static_cast<int>(T), static_cast<int>(D), static_cast<int>(Kkeep), scale
            );
        } else {
            sparse_attn_q1_kernel<__nv_bfloat16, int32_t><<<grid, block, shmem>>>(
                reinterpret_cast<const __nv_bfloat16*>(q2.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(k2.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(v2.data_ptr<at::BFloat16>()),
                idx2.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                BH, static_cast<int>(T), static_cast<int>(D), static_cast<int>(Kkeep), scale
            );
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sparse_attn_q1_kernel: ") + cudaGetErrorString(err));
    }

    // Return [B,H,1,D]
    return out.view({B, H, 1, D});
}


torch::Tensor aq_lut_forward(
    torch::Tensor q,          // [Q, d]
    torch::Tensor codebooks,  // [M, K, d]
    torch::Tensor codes       // [Q, N, M]
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(codebooks.is_cuda(), "codebooks must be a CUDA tensor");
    TORCH_CHECK(codes.is_cuda(), "codes must be a CUDA tensor");
    TORCH_CHECK(q.get_device() == codebooks.get_device(), "q and codebooks must be on the same CUDA device");
    TORCH_CHECK(q.get_device() == codes.get_device(), "q and codes must be on the same CUDA device");

    TORCH_CHECK(q.dim() == 2, "q must be of shape [Q, d]");
    TORCH_CHECK(codebooks.dim() == 3, "codebooks must be of shape [M, K, d]");
    TORCH_CHECK(codes.dim() == 3, "codes must be of shape [Q, N, M]");

    TORCH_CHECK(q.scalar_type() == codebooks.scalar_type(), "q and codebooks must have the same dtype");
    TORCH_CHECK(
        codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kByte || codes.scalar_type() == torch::kChar,
        "codes must be int64 (Long) or uint8 (Byte) or int8 (Char)"
    );

    // Make sure the kernel is launched on the correct GPU in multi-GPU setups.
    const c10::cuda::CUDAGuard device_guard(q.device());

    // Ensure contiguous layout: kernels assume dense contiguous memory.
    // (If inputs are already contiguous, this is a no-op; otherwise PyTorch will create a compact copy.)
    if (!q.is_contiguous()) q = q.contiguous();
    if (!codebooks.is_contiguous()) codebooks = codebooks.contiguous();
    if (!codes.is_contiguous()) codes = codes.contiguous();

    const int Q = static_cast<int>(q.size(0));
    const int d = static_cast<int>(q.size(1));

    const int M = static_cast<int>(codebooks.size(0));
    const int K = static_cast<int>(codebooks.size(1));
    const int d_cb = static_cast<int>(codebooks.size(2));

    TORCH_CHECK(d == d_cb, "q and codebooks last dim must match");

    const int Qc = static_cast<int>(codes.size(0));
    const int N = static_cast<int>(codes.size(1));
    const int Mc = static_cast<int>(codes.size(2));

    TORCH_CHECK(Qc == Q, "codes Q dimension must match q");
    TORCH_CHECK(Mc == M, "codes M dimension must match codebooks");

    auto opts = q.options();
    torch::Tensor scores = torch::zeros({Q, N}, opts);

    const int threads = 256;
    const int blocks_y = (N + threads - 1) / threads;
    dim3 block_dim(threads);
    dim3 grid_dim(Q, blocks_y);

    const auto codes_type = codes.scalar_type();
    if (q.scalar_type() == torch::kFloat16) {
        const size_t shmem = static_cast<size_t>(d) * sizeof(half);
        if (codes_type == torch::kLong) {
            aq_lut_kernel_half<int64_t><<<grid_dim, block_dim, shmem>>>(
                reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(codebooks.data_ptr<at::Half>()),
                codes.data_ptr<int64_t>(),
                reinterpret_cast<half*>(scores.data_ptr<at::Half>()),
                Q, N, M, K, d
            );
        } else if (codes_type == torch::kByte) {
            aq_lut_kernel_half<uint8_t><<<grid_dim, block_dim, shmem>>>(
                reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(codebooks.data_ptr<at::Half>()),
                reinterpret_cast<const uint8_t*>(codes.data_ptr<uint8_t>()),
                reinterpret_cast<half*>(scores.data_ptr<at::Half>()),
                Q, N, M, K, d
            );
        } else {  // torch::kChar
            aq_lut_kernel_half<int8_t><<<grid_dim, block_dim, shmem>>>(
                reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(codebooks.data_ptr<at::Half>()),
                reinterpret_cast<const int8_t*>(codes.data_ptr<int8_t>()),
                reinterpret_cast<half*>(scores.data_ptr<at::Half>()),
                Q, N, M, K, d
            );
        }
    } else {
        AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "aq_lut_kernel", [&] {
            const size_t shmem = static_cast<size_t>(d) * sizeof(scalar_t);
            if (codes_type == torch::kLong) {
                aq_lut_kernel<scalar_t, int64_t><<<grid_dim, block_dim, shmem>>>(
                    q.data_ptr<scalar_t>(),
                    codebooks.data_ptr<scalar_t>(),
                    codes.data_ptr<int64_t>(),
                    scores.data_ptr<scalar_t>(),
                    Q, N, M, K, d
                );
            } else if (codes_type == torch::kByte) {
                aq_lut_kernel<scalar_t, uint8_t><<<grid_dim, block_dim, shmem>>>(
                    q.data_ptr<scalar_t>(),
                    codebooks.data_ptr<scalar_t>(),
                    reinterpret_cast<const uint8_t*>(codes.data_ptr<uint8_t>()),
                    scores.data_ptr<scalar_t>(),
                    Q, N, M, K, d
                );
            } else {  // torch::kChar
                aq_lut_kernel<scalar_t, int8_t><<<grid_dim, block_dim, shmem>>>(
                    q.data_ptr<scalar_t>(),
                    codebooks.data_ptr<scalar_t>(),
                    reinterpret_cast<const int8_t*>(codes.data_ptr<int8_t>()),
                    scores.data_ptr<scalar_t>(),
                    Q, N, M, K, d
                );
            }
        });
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in aq_lut_kernel: ") + cudaGetErrorString(err));
    }

    // Optional: force sync to surface illegal memory access at the right location.
    // Enable by setting env var AQ_LUT_DEBUG_SYNC=1.
    const char* sync_env = std::getenv("AQ_LUT_DEBUG_SYNC");
    if (sync_env != nullptr && sync_env[0] == '1') {
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA sync error in aq_lut_kernel: ") + cudaGetErrorString(err));
        }
    }

    return scores;
}

// ---------------------------------------------------------------------------
// LUT gather kernel: scores[q, n] = sum_m lut[q, m*K + codes[q,n,m]]
// This matches the AQ LUT idea but avoids expensive int64 codes and avoids
// per-n dot products over d.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename code_t>
__global__ void aq_lut_gather_kernel(
    const scalar_t* __restrict__ lut,    // [Q, M*K]
    const code_t* __restrict__ codes,    // [Q, N, M]
    scalar_t* __restrict__ scores,       // [Q, N]
    int Q,
    int N,
    int M,
    int K
) {
    int q_idx = blockIdx.x;
    int n_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (q_idx >= Q || n_idx >= N) return;

    const scalar_t* lut_row = lut + static_cast<long long>(q_idx) * (static_cast<long long>(M) * K);
    const code_t* code_row = codes + (static_cast<long long>(q_idx) * N + n_idx) * M;

    scalar_t acc = static_cast<scalar_t>(0);
    #pragma unroll
    for (int m = 0; m < 32; ++m) {
        if (m >= M) break;
        int code = aq_read_code_as_index<code_t>(code_row[m]);
        if (code < 0) code = 0;
        if (code >= K) code = K - 1;
        acc += lut_row[m * K + code];
    }

    scores[static_cast<long long>(q_idx) * N + n_idx] = acc;
}

template <typename code_t>
__global__ void aq_lut_gather_kernel_half(
    const half* __restrict__ lut,   // [Q, M*K]
    const code_t* __restrict__ codes, // [Q, N, M]
    half* __restrict__ scores,      // [Q, N]
    int Q,
    int N,
    int M,
    int K
) {
    int q_idx = blockIdx.x;
    int n_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (q_idx >= Q || n_idx >= N) return;

    const half* lut_row = lut + static_cast<long long>(q_idx) * (static_cast<long long>(M) * K);
    const code_t* code_row = codes + (static_cast<long long>(q_idx) * N + n_idx) * M;

    float acc = 0.0f;
    #pragma unroll
    for (int m = 0; m < 32; ++m) {
        if (m >= M) break;
        int code = aq_read_code_as_index<code_t>(code_row[m]);
        if (code < 0) code = 0;
        if (code >= K) code = K - 1;
        acc += __half2float(lut_row[m * K + code]);
    }

    scores[static_cast<long long>(q_idx) * N + n_idx] = __float2half_rn(acc);
}

torch::Tensor aq_lut_gather_forward(
    torch::Tensor lut,   // [Q, M*K] (same dtype/device as q)
    torch::Tensor codes  // [Q, N, M]
) {
    TORCH_CHECK(lut.is_cuda(), "lut must be a CUDA tensor");
    TORCH_CHECK(codes.is_cuda(), "codes must be a CUDA tensor");
    TORCH_CHECK(lut.get_device() == codes.get_device(), "lut and codes must be on the same CUDA device");

    TORCH_CHECK(lut.dim() == 2, "lut must be of shape [Q, M*K]");
    TORCH_CHECK(codes.dim() == 3, "codes must be of shape [Q, N, M]");

    TORCH_CHECK(
        lut.scalar_type() == torch::kFloat16 || lut.scalar_type() == torch::kFloat32,
        "lut must be fp16 or fp32"
    );
    TORCH_CHECK(
        codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kByte || codes.scalar_type() == torch::kChar,
        "codes must be int64 (Long) or uint8 (Byte) or int8 (Char)"
    );

    const c10::cuda::CUDAGuard device_guard(lut.device());
    if (!lut.is_contiguous()) lut = lut.contiguous();
    if (!codes.is_contiguous()) codes = codes.contiguous();

    const int Q = static_cast<int>(lut.size(0));
    const int MK = static_cast<int>(lut.size(1));
    const int Qc = static_cast<int>(codes.size(0));
    const int N = static_cast<int>(codes.size(1));
    const int M = static_cast<int>(codes.size(2));
    TORCH_CHECK(Qc == Q, "codes Q dimension must match lut Q");
    TORCH_CHECK(M > 0, "codes M dimension must be > 0");
    TORCH_CHECK(MK % M == 0, "lut second dim must be divisible by M (expect M*K)");
    const int K = MK / M;

    auto opts = lut.options();
    torch::Tensor scores = torch::zeros({Q, N}, opts);

    const int threads = 256;
    const int blocks_y = (N + threads - 1) / threads;
    dim3 block_dim(threads);
    dim3 grid_dim(Q, blocks_y);

    const auto codes_type = codes.scalar_type();
    if (lut.scalar_type() == torch::kFloat16) {
        if (codes_type == torch::kLong) {
            aq_lut_gather_kernel_half<int64_t><<<grid_dim, block_dim>>>(
                reinterpret_cast<const half*>(lut.data_ptr<at::Half>()),
                codes.data_ptr<int64_t>(),
                reinterpret_cast<half*>(scores.data_ptr<at::Half>()),
                Q, N, M, K
            );
        } else if (codes_type == torch::kByte) {
            aq_lut_gather_kernel_half<uint8_t><<<grid_dim, block_dim>>>(
                reinterpret_cast<const half*>(lut.data_ptr<at::Half>()),
                reinterpret_cast<const uint8_t*>(codes.data_ptr<uint8_t>()),
                reinterpret_cast<half*>(scores.data_ptr<at::Half>()),
                Q, N, M, K
            );
        } else {
            aq_lut_gather_kernel_half<int8_t><<<grid_dim, block_dim>>>(
                reinterpret_cast<const half*>(lut.data_ptr<at::Half>()),
                reinterpret_cast<const int8_t*>(codes.data_ptr<int8_t>()),
                reinterpret_cast<half*>(scores.data_ptr<at::Half>()),
                Q, N, M, K
            );
        }
    } else {
        // fp32
        if (codes_type == torch::kLong) {
            aq_lut_gather_kernel<float, int64_t><<<grid_dim, block_dim>>>(
                lut.data_ptr<float>(),
                codes.data_ptr<int64_t>(),
                scores.data_ptr<float>(),
                Q, N, M, K
            );
        } else if (codes_type == torch::kByte) {
            aq_lut_gather_kernel<float, uint8_t><<<grid_dim, block_dim>>>(
                lut.data_ptr<float>(),
                reinterpret_cast<const uint8_t*>(codes.data_ptr<uint8_t>()),
                scores.data_ptr<float>(),
                Q, N, M, K
            );
        } else {
            aq_lut_gather_kernel<float, int8_t><<<grid_dim, block_dim>>>(
                lut.data_ptr<float>(),
                reinterpret_cast<const int8_t*>(codes.data_ptr<int8_t>()),
                scores.data_ptr<float>(),
                Q, N, M, K
            );
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in aq_lut_gather_kernel: ") + cudaGetErrorString(err));
    }
    return scores;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aq_lut_forward", &aq_lut_forward, "AQ LUT-based approximate inner product (CUDA)");
    m.def("aq_lut_gather_forward", &aq_lut_gather_forward, "AQ LUT gather+sum (CUDA)");
    m.def("sparse_attn_q1_forward", &sparse_attn_q1_forward, "Sparse/pruned attention forward for decode (Q=1) (CUDA)");
}


