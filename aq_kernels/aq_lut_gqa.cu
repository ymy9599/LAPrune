/*
 * aq_lut_gather_v2.cu — GQA-aware LUT Gather for AQ decode
 *
 * Supports two modes:
 *   Dense:  scores[h,t] = Σ_m lut[h, m, codes[h/G, t, m]]   (coarse scoring)
 *   Sparse: scores[h,n] = Σ_m lut[h, m, codes[h/G, idx[h,n], m]] (fine scoring)
 *
 * Key optimisation: LUT for each head is loaded into shared memory once and
 * reused across all tokens/kept-positions — eliminates redundant global reads.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

template <typename idx_t>
__global__ void lut_gather_kernel(
    const __half* __restrict__ lut,      // [H, M_stride, K]
    const uint8_t* __restrict__ codes,   // [Hkv, T, M_stride]
    const idx_t*   __restrict__ indices, // [H, N]  (NULL → dense)
    __half* __restrict__ out,            // [H, N]
    const int Hkv,  const int G,
    const int T,    const int N,
    const int M_start, const int M_range,
    const int M_stride, const int K,
    const int64_t codes_stride_h,
    const int64_t codes_stride_t)
{
    extern __shared__ __half s_lut[];          // M_range × K

    const int h   = blockIdx.x;
    const int hkv = h / G;

    /* ── cooperative LUT load ── */
    const int lut_elems = M_range * K;
    const __half* src = lut + (long long)h * M_stride * K
                            + (long long)M_start * K;
    for (int i = threadIdx.x; i < lut_elems; i += blockDim.x)
        s_lut[i] = src[i];
    __syncthreads();

    /* ── per-token scoring ── */
    for (int n = (int)blockIdx.y * blockDim.x + threadIdx.x;
         n < N;
         n += (int)gridDim.y * blockDim.x)
    {
        int t;
        if (indices) {
            t = (int)indices[(long long)h * N + n];
            if (t < 0 || t >= T) {
                out[(long long)h * N + n] = __float2half(0.f);
                continue;
            }
        } else {
            t = n;
        }

        // The decode code cache is preallocated with extra sequence capacity.
        // Its valid [:T] prefix is therefore not contiguous across heads. Read
        // it using the real tensor strides instead of materialising a 32 MiB
        // contiguous copy for every layer and every decode token.
        const uint8_t* cp = codes + (long long)hkv * codes_stride_h
                                   + (long long)t * codes_stride_t
                                   + M_start;
        float acc = 0.f;
        #pragma unroll 8
        for (int m = 0; m < M_range; m++)
            acc += __half2float(s_lut[m * K + cp[m]]);

        out[(long long)h * N + n] = __float2half(acc);
    }
}

/* ── legacy flat interface (codes already expanded to H) ── */
__global__ void lut_gather_flat_kernel(
    const __half* __restrict__ lut,    // [BH, M, K]
    const uint8_t* __restrict__ codes, // [BH, T, M]
    __half* __restrict__ out,          // [BH, T]
    const int BH, const int T,
    const int M,  const int K)
{
    extern __shared__ __half s_lut[];  // M × K

    const int bh = blockIdx.x;
    const int lut_elems = M * K;
    const __half* src = lut + (long long)bh * M * K;
    for (int i = threadIdx.x; i < lut_elems; i += blockDim.x)
        s_lut[i] = src[i];
    __syncthreads();

    for (int t = (int)blockIdx.y * blockDim.x + threadIdx.x;
         t < T;
         t += (int)gridDim.y * blockDim.x)
    {
        const uint8_t* cp = codes + ((long long)bh * T + t) * M;
        float acc = 0.f;
        #pragma unroll 8
        for (int m = 0; m < M; m++)
            acc += __half2float(s_lut[m * K + cp[m]]);
        out[(long long)bh * T + t] = __float2half(acc);
    }
}

/* Score all query heads in one GQA group while loading each KV-head code only
 * once. The per-head accumulation order over M is identical to the original
 * kernel, so fp16 outputs remain bit-exact. */
__global__ void lut_gather_grouped_kernel(
    const __half* __restrict__ lut,      // [Hkv*G,M,K]
    const uint8_t* __restrict__ codes,   // [Hkv,T,M], possibly strided in H
    __half* __restrict__ out,            // [Hkv*G,T]
    const int Hkv, const int G, const int T,
    const int M, const int K,
    const int64_t codes_stride_h,
    const int64_t codes_stride_t)
{
    extern __shared__ __half s_lut[];    // [G,M,K]
    const int hkv = (int)blockIdx.x;
    const int group_lut_elems = G * M * K;
    const __half* lut_src = lut + (int64_t)hkv * G * M * K;
    for (int i = (int)threadIdx.x; i < group_lut_elems; i += (int)blockDim.x)
        s_lut[i] = lut_src[i];
    __syncthreads();

    for (int t = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
         t < T;
         t += (int)gridDim.y * (int)blockDim.x)
    {
        const uint8_t* cp = codes + (int64_t)hkv * codes_stride_h
                                   + (int64_t)t * codes_stride_t;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int m = 0; m < M; ++m) {
            const int code = (int)cp[m];
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                if (g < G)
                    acc[g] += __half2float(s_lut[(g * M + m) * K + code]);
            }
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            if (g < G)
                out[((int64_t)hkv * G + g) * T + t] = __float2half(acc[g]);
        }
    }
}

/* Specialized Llama-3.1 GQA path.  The generic grouped kernel stores the LUT
 * as [G,M,K], so one code requires four distant shared-memory lookups.  This
 * kernel transposes it cooperatively to [M,K,G]; all four query-head values
 * for the same code then occupy one aligned 8-byte slot and are consumed as
 * two half2 loads.  The FP32 accumulation order within each head is unchanged,
 * so output is bit-exact with lut_gather_grouped_kernel. */
template <int BLOCK_THREADS>
__global__ void lut_gather_grouped_interleaved_kernel(
    const __half* __restrict__ lut,      // [Hkv*4,32,256]
    const uint8_t* __restrict__ codes,   // [Hkv,T,32], possibly strided in H
    __half* __restrict__ out,            // [Hkv*4,T]
    const int Hkv, const int T,
    const int64_t codes_stride_h,
    const int64_t codes_stride_t)
{
    constexpr int G = 4;
    constexpr int M = 32;
    constexpr int K = 256;
    extern __shared__ __half s_lut[];    // [M,K,G]
    const int hkv = (int)blockIdx.x;
    const __half* src = lut + (int64_t)hkv * G * M * K;

    for (int source = (int)threadIdx.x; source < G * M * K;
         source += BLOCK_THREADS) {
        const int g = source / (M * K);
        const int mk = source - g * M * K;
        s_lut[mk * G + g] = src[source];
    }
    __syncthreads();

    for (int t = (int)blockIdx.y * BLOCK_THREADS + (int)threadIdx.x;
         t < T;
         t += (int)gridDim.y * BLOCK_THREADS) {
        const uint8_t* cp = codes + (int64_t)hkv * codes_stride_h
                                   + (int64_t)t * codes_stride_t;
        // M=32 codes occupy one aligned 32-byte row. Fetch the complete row
        // with two 16-byte transactions, then extract bytes from registers;
        // this avoids 32 byte-load instructions with a 32-byte warp stride.
        const uint4* code_vec = reinterpret_cast<const uint4*>(cp);
        const uint4 c0 = code_vec[0];
        const uint4 c1 = code_vec[1];
        const uint32_t code_words[8] = {
            c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w
        };
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const int code = (int)((code_words[m >> 2] >> ((m & 3) * 8)) & 0xffu);
            const __half* packed = s_lut + (m * K + code) * G;
            const __half2 pair01 = *reinterpret_cast<const __half2*>(packed);
            const __half2 pair23 = *reinterpret_cast<const __half2*>(packed + 2);
            acc0 += __half2float(__low2half(pair01));
            acc1 += __half2float(__high2half(pair01));
            acc2 += __half2float(__low2half(pair23));
            acc3 += __half2float(__high2half(pair23));
        }
        out[((int64_t)hkv * G + 0) * T + t] = __float2half(acc0);
        out[((int64_t)hkv * G + 1) * T + t] = __float2half(acc1);
        out[((int64_t)hkv * G + 2) * T + t] = __float2half(acc2);
        out[((int64_t)hkv * G + 3) * T + t] = __float2half(acc3);
    }
}

/* ── Native-GQA K/V gather ────────────────────────────────────────────────
 *
 * torch.take_along_dim normalises every index with a full-sized remainder
 * kernel before copying K and V. LAPrune's Top-K indices are already known to
 * be non-negative and in range, so that work is unnecessary. This kernel also
 * copies K and V in one launch and reads the preallocated, strided cache
 * directly.
 */
__global__ void gqa_gather_kv_half2_kernel(
    const __half* __restrict__ keys,       // [B,Hkv,T,D], possibly strided in H
    const __half* __restrict__ vals,       // [B,Hkv,T,D], possibly strided in H
    const int64_t* __restrict__ indices,   // [B,Hq,N], contiguous
    __half* __restrict__ out_k,            // [B,Hq,N,D], contiguous
    __half* __restrict__ out_v,            // [B,Hq,N,D], contiguous
    const int B, const int Hkv, const int Hq,
    const int T, const int N, const int D,
    const int G, const int tokens_per_block,
    const int explicit_N, const int implicit_tail_start,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_t,
    const int64_t v_stride_b, const int64_t v_stride_h,
    const int64_t v_stride_t)
{
    const int bh = (int)blockIdx.x;
    const int b = bh / Hq;
    const int hq = bh - b * Hq;
    const int hkv = hq / G;
    const int tid = (int)threadIdx.x;
    const int d2_count = D >> 1;

    __shared__ int64_t selected[16];
    if (tid < tokens_per_block) {
        const int n = (int)blockIdx.y * tokens_per_block + tid;
        if (n < explicit_N) {
            selected[tid] = indices[((int64_t)b * Hq + hq) * explicit_N + n];
        } else if (n < N && implicit_tail_start >= 0) {
            selected[tid] = (int64_t)implicit_tail_start + (n - explicit_N);
        } else {
            selected[tid] = -1;
        }
    }
    __syncthreads();

    const int slot = tid / d2_count;
    const int d2 = tid - slot * d2_count;
    if (slot >= tokens_per_block) return;
    const int n = (int)blockIdx.y * tokens_per_block + slot;
    if (n >= N) return;
    const int64_t t = selected[slot];
    if (t < 0 || t >= T) return;

    const int64_t k_elem = (int64_t)b * k_stride_b + (int64_t)hkv * k_stride_h
                         + t * k_stride_t + (int64_t)d2 * 2;
    const int64_t v_elem = (int64_t)b * v_stride_b + (int64_t)hkv * v_stride_h
                         + t * v_stride_t + (int64_t)d2 * 2;
    const int64_t out_half2 = (((int64_t)b * Hq + hq) * N + n) * d2_count + d2;

    reinterpret_cast<__half2*>(out_k)[out_half2] =
        *reinterpret_cast<const __half2*>(keys + k_elem);
    reinterpret_cast<__half2*>(out_v)[out_half2] =
        *reinterpret_cast<const __half2*>(vals + v_elem);
}

__global__ void gqa_gather_kv_half8_kernel(
    const __half* __restrict__ keys, const __half* __restrict__ vals,
    const int64_t* __restrict__ indices,
    __half* __restrict__ out_k, __half* __restrict__ out_v,
    const int B, const int Hkv, const int Hq,
    const int T, const int N, const int D,
    const int G, const int tokens_per_block,
    const int explicit_N, const int implicit_tail_start,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_t,
    const int64_t v_stride_b, const int64_t v_stride_h,
    const int64_t v_stride_t)
{
    const int bh = (int)blockIdx.x;
    const int b = bh / Hq;
    const int hq = bh - b * Hq;
    const int hkv = hq / G;
    const int tid = (int)threadIdx.x;
    const int d8_count = D / 8;
    __shared__ int64_t selected[16];
    if (tid < tokens_per_block) {
        const int n = (int)blockIdx.y * tokens_per_block + tid;
        if (n < explicit_N)
            selected[tid] = indices[((int64_t)b * Hq + hq) * explicit_N + n];
        else if (n < N && implicit_tail_start >= 0)
            selected[tid] = (int64_t)implicit_tail_start + (n - explicit_N);
        else
            selected[tid] = -1;
    }
    __syncthreads();
    const int slot = tid / d8_count;
    const int d8 = tid - slot * d8_count;
    if (slot >= tokens_per_block) return;
    const int n = (int)blockIdx.y * tokens_per_block + slot;
    if (n >= N) return;
    const int64_t t = selected[slot];
    if (t < 0 || t >= T) return;
    const int64_t k_elem = (int64_t)b * k_stride_b + (int64_t)hkv * k_stride_h
                         + t * k_stride_t + (int64_t)d8 * 8;
    const int64_t v_elem = (int64_t)b * v_stride_b + (int64_t)hkv * v_stride_h
                         + t * v_stride_t + (int64_t)d8 * 8;
    const int64_t out_vec = (((int64_t)b * Hq + hq) * N + n) * d8_count + d8;
    reinterpret_cast<int4*>(out_k)[out_vec] =
        *reinterpret_cast<const int4*>(keys + k_elem);
    reinterpret_cast<int4*>(out_v)[out_vec] =
        *reinterpret_cast<const int4*>(vals + v_elem);
}

__device__ __forceinline__ __half laprune_hmul_rn(__half a, __half b) {
    unsigned short av = *reinterpret_cast<unsigned short*>(&a);
    unsigned short bv = *reinterpret_cast<unsigned short*>(&b);
    unsigned short ov;
    asm("mul.rn.f16 %0, %1, %2;" : "=h"(ov) : "h"(av), "h"(bv));
    return *reinterpret_cast<__half*>(&ov);
}

__device__ __forceinline__ __half laprune_hadd_rn(__half a, __half b) {
    unsigned short av = *reinterpret_cast<unsigned short*>(&a);
    unsigned short bv = *reinterpret_cast<unsigned short*>(&b);
    unsigned short ov;
    asm("add.rn.f16 %0, %1, %2;" : "=h"(ov) : "h"(av), "h"(bv));
    return *reinterpret_cast<__half*>(&ov);
}

/* Apply HF's half-precision NeoX/Llama RoPE arithmetic to Q and K in one
 * launch. cos/sin are the official model-generated tensors, so this changes
 * only launch structure, not the rotary frequencies or scaling policy. */




/* ═══════════════════════  C++ / Python interface  ═══════════════════════ */

torch::Tensor aq_lut_gather(
    torch::Tensor lut,      // [H, M_stride, K] float16
    torch::Tensor codes,    // [Hkv, T, M_stride] uint8
    int G,
    int M_start,
    int M_end,
    c10::optional<torch::Tensor> indices)   // [H, N] int64 | None
{
    TORCH_CHECK(lut.is_cuda() && codes.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(lut.scalar_type() == torch::kFloat16, "lut must be fp16");
    TORCH_CHECK(codes.scalar_type() == torch::kByte, "codes must be uint8");
    TORCH_CHECK(lut.dim() == 3 && codes.dim() == 3);

    auto lut_c = lut.contiguous();
    // Keep a possibly non-contiguous sequence prefix. The kernel consumes its
    // actual head/token strides; M must remain the innermost contiguous axis.
    auto codes_c = codes;
    TORCH_CHECK(codes_c.stride(2) == 1,
                "codes M dimension must be contiguous");

    const int H        = (int)lut_c.size(0);
    const int M_stride = (int)lut_c.size(1);
    const int K        = (int)lut_c.size(2);
    const int Hkv      = (int)codes_c.size(0);
    const int T        = (int)codes_c.size(1);
    const int M_range  = M_end - M_start;
    const int64_t codes_stride_h = codes_c.stride(0);
    const int64_t codes_stride_t = codes_c.stride(1);

    TORCH_CHECK(H == Hkv * G, "H must equal Hkv * G");
    TORCH_CHECK(M_end <= M_stride && M_start >= 0 && M_range > 0);

    bool has_idx = indices.has_value() && indices.value().defined();
    int N = T;
    torch::Tensor idx_c;
    if (has_idx) {
        idx_c = indices.value().contiguous().to(torch::kLong);
        TORCH_CHECK(idx_c.size(0) == H);
        N = (int)idx_c.size(1);
    }

    auto out = torch::empty({H, N}, lut_c.options());
    if (N == 0) return out;

    c10::cuda::CUDAGuard device_guard(lut_c.device());

    const int BLOCK = 1024;
    int grid_y = (N + BLOCK - 1) / BLOCK;
    grid_y = std::min(grid_y, 2048);
    dim3 grid(H, grid_y);
    size_t smem = (size_t)M_range * K * sizeof(__half);

    auto stream = c10::cuda::getCurrentCUDAStream(lut_c.device().index());

    if (has_idx) {
        lut_gather_kernel<int64_t><<<grid, BLOCK, smem, stream>>>(
            reinterpret_cast<const __half*>(lut_c.data_ptr<at::Half>()),
            codes_c.data_ptr<uint8_t>(),
            idx_c.data_ptr<int64_t>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            Hkv, G, T, N, M_start, M_range, M_stride, K,
            codes_stride_h, codes_stride_t);
    } else {
        lut_gather_kernel<int64_t><<<grid, BLOCK, smem, stream>>>(
            reinterpret_cast<const __half*>(lut_c.data_ptr<at::Half>()),
            codes_c.data_ptr<uint8_t>(),
            (const int64_t*)nullptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            Hkv, G, T, N, M_start, M_range, M_stride, K,
            codes_stride_h, codes_stride_t);
    }
    return out;
}

torch::Tensor aq_lut_gather_flat(
    torch::Tensor lut_flat,   // [BH, M*K] float16
    torch::Tensor codes_flat) // [BH, T, M] uint8
{
    TORCH_CHECK(lut_flat.is_cuda() && codes_flat.is_cuda());
    TORCH_CHECK(lut_flat.scalar_type() == torch::kFloat16);
    TORCH_CHECK(codes_flat.scalar_type() == torch::kByte);

    auto lut_c   = lut_flat.contiguous();
    auto codes_c = codes_flat.contiguous();

    const int BH = (int)lut_c.size(0);
    const int M  = (int)codes_c.size(2);
    const int K  = (int)lut_c.size(1) / M;
    const int T  = (int)codes_c.size(1);

    auto lut_3d = lut_c.reshape({BH, M, K});
    auto out    = torch::empty({BH, T}, lut_c.options());
    if (T == 0) return out;

    c10::cuda::CUDAGuard device_guard(lut_3d.device());

    const int BLOCK = 1024;
    int grid_y = std::min((T + BLOCK - 1) / BLOCK, 2048);
    dim3 grid(BH, grid_y);
    size_t smem = (size_t)M * K * sizeof(__half);

    auto stream = c10::cuda::getCurrentCUDAStream(lut_3d.device().index());
    lut_gather_flat_kernel<<<grid, BLOCK, smem, stream>>>(
        reinterpret_cast<const __half*>(lut_3d.data_ptr<at::Half>()),
        codes_c.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        BH, T, M, K);
    return out;
}

torch::Tensor aq_lut_gather_grouped(
    torch::Tensor lut, torch::Tensor codes, int G, int grid_y_cap)
{
    TORCH_CHECK(lut.is_cuda() && codes.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(lut.scalar_type() == torch::kFloat16 &&
                codes.scalar_type() == torch::kByte,
                "grouped LUT requires fp16 LUT and uint8 codes");
    TORCH_CHECK(lut.dim() == 3 && codes.dim() == 3);
    TORCH_CHECK(codes.stride(2) == 1, "codes M dimension must be contiguous");

    auto lut_c = lut.contiguous();
    const int H = (int)lut_c.size(0);
    const int M = (int)lut_c.size(1);
    const int K = (int)lut_c.size(2);
    const int Hkv = (int)codes.size(0);
    const int T = (int)codes.size(1);
    TORCH_CHECK(H == Hkv * G, "H must equal Hkv*G");
    TORCH_CHECK(G > 0 && G <= 4, "grouped kernel supports G<=4");

    auto out = torch::empty({H, T}, lut_c.options());
    if (T == 0) return out;
    TORCH_CHECK(grid_y_cap > 0, "grid_y_cap must be positive");
    const size_t smem = (size_t)G * M * K * sizeof(__half);
    TORCH_CHECK(smem <= 101376, "grouped LUT shared-memory request is too large");

    c10::cuda::CUDAGuard device_guard(lut_c.device());
    const int dev = lut_c.device().index();
    auto stream = c10::cuda::getCurrentCUDAStream(dev);

    if (G == 4 && M == 32 && K == 256) {
        constexpr int BLOCK = 256;
        // Across 4K--128K, 16 persistent token blocks per KV head gave the
        // best or tied-best latency on RTX 6000D while remaining bit-exact.
        const int grid_y = std::min((T + BLOCK - 1) / BLOCK, 16);
        static bool interleaved_attr_set[16] = {false};
        if (dev >= 0 && dev < 16 && !interleaved_attr_set[dev]) {
            auto err = cudaFuncSetAttribute(
                lut_gather_grouped_interleaved_kernel<BLOCK>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)smem);
            TORCH_CHECK(err == cudaSuccess,
                        "failed to opt in interleaved LUT dynamic shared memory: ",
                        cudaGetErrorString(err));
            interleaved_attr_set[dev] = true;
        }
        dim3 grid(Hkv, grid_y);
        lut_gather_grouped_interleaved_kernel<BLOCK><<<grid, BLOCK, smem, stream>>>(
            reinterpret_cast<const __half*>(lut_c.data_ptr<at::Half>()),
            codes.data_ptr<uint8_t>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            Hkv, T, codes.stride(0), codes.stride(1));
        return out;
    }

    constexpr int BLOCK = 1024;
    const int grid_y = std::min((T + BLOCK - 1) / BLOCK, grid_y_cap);
    static bool attr_set[16] = {false};
    if (dev >= 0 && dev < 16 && !attr_set[dev]) {
        auto err = cudaFuncSetAttribute(
            lut_gather_grouped_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        TORCH_CHECK(err == cudaSuccess,
                    "failed to opt in grouped LUT dynamic shared memory: ",
                    cudaGetErrorString(err));
        attr_set[dev] = true;
    }
    dim3 grid(Hkv, grid_y);
    lut_gather_grouped_kernel<<<grid, BLOCK, smem, stream>>>(
        reinterpret_cast<const __half*>(lut_c.data_ptr<at::Half>()),
        codes.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        Hkv, G, T, M, K, codes.stride(0), codes.stride(1));
    return out;
}

std::vector<torch::Tensor> gqa_gather_kv(
    torch::Tensor keys,       // [B,Hkv,T,D] fp16
    torch::Tensor vals,       // [B,Hkv,T,D] fp16
    torch::Tensor indices,    // [B,Hq,N] int64
    int G, int threads, int vector_width)
{
    TORCH_CHECK(keys.is_cuda() && vals.is_cuda() && indices.is_cuda(),
                "keys, vals and indices must be CUDA tensors");
    TORCH_CHECK(keys.scalar_type() == torch::kFloat16 &&
                vals.scalar_type() == torch::kFloat16,
                "gqa_gather_kv currently supports fp16 K/V");
    TORCH_CHECK(indices.scalar_type() == torch::kLong,
                "indices must be int64");
    TORCH_CHECK(keys.dim() == 4 && vals.dim() == 4 && indices.dim() == 3,
                "expected K/V [B,Hkv,T,D] and indices [B,Hq,N]");
    TORCH_CHECK(keys.sizes() == vals.sizes(), "K/V shapes must match");
    TORCH_CHECK(keys.stride(3) == 1 && vals.stride(3) == 1,
                "K/V head dimension must be contiguous");

    const int B = (int)keys.size(0);
    const int Hkv = (int)keys.size(1);
    const int T = (int)keys.size(2);
    const int D = (int)keys.size(3);
    const int Hq = (int)indices.size(1);
    const int N = (int)indices.size(2);
    TORCH_CHECK((int)indices.size(0) == B, "indices batch mismatch");
    TORCH_CHECK(Hq == Hkv * G, "Hq must equal Hkv * G");
    TORCH_CHECK(D > 0 && D % 2 == 0, "D must be positive and even");

    auto idx = indices.contiguous();
    auto out_k = torch::empty({B, Hq, N, D}, keys.options());
    auto out_v = torch::empty({B, Hq, N, D}, vals.options());
    if (N == 0) return {out_k, out_v};

    TORCH_CHECK(vector_width == 2 || vector_width == 8,
                "vector_width must be 2 or 8 half elements");
    TORCH_CHECK(D % vector_width == 0, "D must be divisible by vector_width");
    const int dvec_count = D / vector_width;
    TORCH_CHECK((threads == 256 || threads == 512 || threads == 1024) &&
                dvec_count <= threads && threads % dvec_count == 0,
                "threads must be 256/512/1024 and divisible by D/vector_width");
    const int tokens_per_block = threads / dvec_count;
    TORCH_CHECK(tokens_per_block <= 16, "internal shared index capacity exceeded");

    c10::cuda::CUDAGuard device_guard(keys.device());
    auto stream = c10::cuda::getCurrentCUDAStream(keys.device().index());
    dim3 grid(B * Hq, (N + tokens_per_block - 1) / tokens_per_block);
    if (vector_width == 8) {
        gqa_gather_kv_half8_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(keys.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(vals.data_ptr<at::Half>()),
        idx.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(out_k.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out_v.data_ptr<at::Half>()),
        B, Hkv, Hq, T, N, D, G, tokens_per_block, N, -1,
        keys.stride(0), keys.stride(1), keys.stride(2),
        vals.stride(0), vals.stride(1), vals.stride(2));
    } else gqa_gather_kv_half2_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(keys.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(vals.data_ptr<at::Half>()),
        idx.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(out_k.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out_v.data_ptr<at::Half>()),
        B, Hkv, Hq, T, N, D, G, tokens_per_block,
        N, -1,
        keys.stride(0), keys.stride(1), keys.stride(2),
        vals.stride(0), vals.stride(1), vals.stride(2));
    return {out_k, out_v};
}

std::vector<torch::Tensor> gqa_gather_kv_with_tail(
    torch::Tensor keys, torch::Tensor vals, torch::Tensor indices,
    int G, int tail_start, int tail_count, int threads, int vector_width)
{
    TORCH_CHECK(keys.is_cuda() && vals.is_cuda() && indices.is_cuda(),
                "keys, vals and indices must be CUDA tensors");
    TORCH_CHECK(keys.scalar_type() == torch::kFloat16 &&
                vals.scalar_type() == torch::kFloat16 &&
                indices.scalar_type() == torch::kLong,
                "gqa_gather_kv_with_tail requires fp16 K/V and int64 indices");
    TORCH_CHECK(keys.dim() == 4 && vals.dim() == 4 && indices.dim() == 3,
                "expected K/V [B,Hkv,T,D], indices [B,Hq,N]");
    TORCH_CHECK(keys.sizes() == vals.sizes(), "K/V shapes must match");
    const int B = (int)keys.size(0), Hkv = (int)keys.size(1);
    const int T = (int)keys.size(2), D = (int)keys.size(3);
    const int Hq = (int)indices.size(1), explicit_N = (int)indices.size(2);
    TORCH_CHECK((int)indices.size(0) == B && Hq == Hkv * G, "GQA shape mismatch");
    TORCH_CHECK(tail_count >= 0 && tail_start >= 0 && tail_start + tail_count <= T,
                "invalid implicit tail range");
    TORCH_CHECK(D > 0 && D % 2 == 0 && keys.stride(3) == 1 && vals.stride(3) == 1,
                "D must be even and contiguous");
    const int N = explicit_N + tail_count;
    auto idx = indices.contiguous();
    auto out_k = torch::empty({B, Hq, N, D}, keys.options());
    auto out_v = torch::empty({B, Hq, N, D}, vals.options());
    if (N == 0) return {out_k, out_v};
    TORCH_CHECK(vector_width == 2 || vector_width == 8,
                "vector_width must be 2 or 8 half elements");
    TORCH_CHECK(D % vector_width == 0, "D must be divisible by vector_width");
    const int dvec_count = D / vector_width;
    TORCH_CHECK((threads == 256 || threads == 512 || threads == 1024) &&
                dvec_count <= threads && threads % dvec_count == 0,
                "threads must be 256/512/1024 and divisible by D/vector_width");
    const int tokens_per_block = threads / dvec_count;
    c10::cuda::CUDAGuard device_guard(keys.device());
    auto stream = c10::cuda::getCurrentCUDAStream(keys.device().index());
    dim3 grid(B * Hq, (N + tokens_per_block - 1) / tokens_per_block);
    if (vector_width == 8) {
        gqa_gather_kv_half8_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(keys.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(vals.data_ptr<at::Half>()),
        idx.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(out_k.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out_v.data_ptr<at::Half>()),
        B, Hkv, Hq, T, N, D, G, tokens_per_block, explicit_N, tail_start,
        keys.stride(0), keys.stride(1), keys.stride(2),
        vals.stride(0), vals.stride(1), vals.stride(2));
    } else gqa_gather_kv_half2_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(keys.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(vals.data_ptr<at::Half>()),
        idx.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(out_k.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out_v.data_ptr<at::Half>()),
        B, Hkv, Hq, T, N, D, G, tokens_per_block,
        explicit_N, tail_start,
        keys.stride(0), keys.stride(1), keys.stride(2),
        vals.stride(0), vals.stride(1), vals.stride(2));
    return {out_k, out_v};
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aq_lut_gather", &aq_lut_gather,
          "GQA-aware LUT gather (dense/sparse)",
          py::arg("lut"), py::arg("codes"), py::arg("G"),
          py::arg("M_start"), py::arg("M_end"),
          py::arg("indices") = py::none());
    m.def("aq_lut_gather_flat", &aq_lut_gather_flat,
          "Legacy flat LUT gather (drop-in for old _aq_lut_gather_cuda)");
    m.def("gqa_gather_kv", &gqa_gather_kv,
          "Fused native-GQA K/V gather without index remainder",
          py::arg("keys"), py::arg("vals"), py::arg("indices"), py::arg("G"),
          py::arg("threads") = 256, py::arg("vector_width") = 8);
    m.def("gqa_gather_kv_with_tail", &gqa_gather_kv_with_tail,
          "Fused native-GQA gather with an implicit contiguous local tail",
          py::arg("keys"), py::arg("vals"), py::arg("indices"), py::arg("G"),
          py::arg("tail_start"), py::arg("tail_count"), py::arg("threads") = 256,
          py::arg("vector_width") = 8);
    m.def("aq_lut_gather_grouped", &aq_lut_gather_grouped,
          "GQA-grouped dense AQ LUT scan",
          py::arg("lut"), py::arg("codes"), py::arg("G"),
          py::arg("grid_y_cap") = 48);
}
