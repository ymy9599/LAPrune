import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import torch

from ..quantizer import AdditiveQuantizer

try:
    # Prefer vendored kernels under LAPrune/aq_kernels (open-source packaging),
    # fallback to top-level `aq_kernels` for backwards compatibility.
    try:
        from ...aq_kernels import aq_lut_forward as aq_lut_cuda, aq_lut_gather_forward as aq_lut_gather_cuda  # type: ignore
    except Exception:
        from aq_kernels import aq_lut_forward as aq_lut_cuda, aq_lut_gather_forward as aq_lut_gather_cuda  # type: ignore
    HAS_AQ_LUT_CUDA = aq_lut_cuda is not None and aq_lut_gather_cuda is not None
except Exception:
    aq_lut_cuda = None
    aq_lut_gather_cuda = None
    HAS_AQ_LUT_CUDA = False

try:
    try:
        from ...aq_kernels import aq_lut_gather_gqa, aq_lut_gather_grouped  # type: ignore
    except Exception:
        from aq_kernels import aq_lut_gather_gqa, aq_lut_gather_grouped  # type: ignore
    HAS_AQ_LUT_GQA_CUDA = aq_lut_gather_gqa is not None
except Exception:
    aq_lut_gather_gqa = None
    aq_lut_gather_grouped = None
    HAS_AQ_LUT_GQA_CUDA = False

try:
    try:
        from ...aq_kernels import fused_linear_encode_forward as aq_encode_cuda  # type: ignore
    except Exception:
        from aq_kernels import fused_linear_encode_forward as aq_encode_cuda  # type: ignore
    HAS_AQ_ENCODE_CUDA = aq_encode_cuda is not None
except Exception:
    aq_encode_cuda = None
    HAS_AQ_ENCODE_CUDA = False

USE_AQ_LUT_CUDA = True

def _fast_greedy_encode(aq: AdditiveQuantizer, x: torch.Tensor) -> Optional[torch.Tensor]:
    """CUDA fast path for greedy AQ encode (icm_iters=0).

    The kernel solves the same greedy nearest-codeword step as the PyTorch path:
    argmin ||r - c||^2 is equivalent to argmax dot(r, c) - 0.5 * ||c||^2.
    """
    if (
        aq_encode_cuda is None
        or aq.codebooks is None
        or not x.is_cuda
        or aq.K > 256
        or x.dtype not in (torch.float16, torch.float32)
        or aq.codebooks.dtype not in (torch.float16, torch.float32)
    ):
        return None

    x_h = x.to(dtype=torch.float16).contiguous()
    cache_key = (
        int(aq.codebooks.data_ptr()),
        str(aq.codebooks.device),
        str(aq.codebooks.dtype),
        str(x_h.device),
    )
    cached = getattr(aq, "_fast_encode_cache", None)
    if not isinstance(cached, tuple) or cached[0] != cache_key:
        codebooks_h = aq.codebooks.to(device=x_h.device, dtype=torch.float16).contiguous()
        weight_t = codebooks_h.transpose(1, 2).contiguous()
        bias = (-0.5 * (codebooks_h.float().pow(2).sum(dim=-1))).to(dtype=torch.float16).contiguous()
        aq._fast_encode_cache = (cache_key, weight_t, bias, codebooks_h)
    else:
        _, weight_t, bias, codebooks_h = cached
    return aq_encode_cuda(x_h, weight_t, bias, codebooks_h, True, 0)


@dataclass
class AQLayerConfig:
    """
    AQ codebook configuration for one layer:
    - M: number of sub-codebooks
    - K: number of entries per sub-codebook
    - d: vector dimension (per-head head_dim)
    - heads_per_group: how many heads share one codebook (e.g. 2 => 2 heads share)
    """
    M: int
    K: int
    d: int
    heads_per_group: int = 2
    # Optional metadata for safer loading:
    # when provided, we can compute how many groups should exist and skip stale group files.
    num_heads: Optional[int] = None


class AQLayerCodebookManager:
    """
    Manage AQ codebooks (by layer / head-group) and approximate inner-product scoring.

    Design goals:
    - independent from other baselines / legacy code paths;
    - per-layer codebooks;
    - head-group sharing (e.g. 2 heads share one codebook);
    - persist codebooks on disk; inference loads read-only;
    - LUT-based approx inner product for one query:
      s_i ≈ sum_m (q · C_m[z_{i,m}]) for ranking / key selection.
    """

    def __init__(
        self,
        layer_cfgs: Dict[int, AQLayerConfig],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Args:
        - layer_cfgs: {layer_idx: AQLayerConfig}, per-layer M/K/d etc.
        - device/dtype: device and dtype for AQ codebooks (training + inference)
        """
        self.layer_cfgs = layer_cfgs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # (layer_idx, group_idx) -> AdditiveQuantizer instance
        self._aq_per_group: Dict[Tuple[int, int], AdditiveQuantizer] = {}

    # ------------------------------------------------------------------
    # Training APIs
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_layer(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        n_iter: int = 10,
        batch_size: int = 3_000_000,
        ema_decay: Optional[float] = None,
        max_head_groups: Optional[int] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Train AQ codebooks for one layer.

        Args:
        - layer_idx: layer index
        - keys: sampled keys with shape [N, num_heads, head_dim]
        - n_iter / batch_size / ema_decay: forwarded to AdditiveQuantizer.fit

        Convention:
        - Within one layer, every `heads_per_group` heads share one codebook.
          Example: num_heads=16, heads_per_group=2 => group_idx=0..7 (2 heads per group).
        """
        assert layer_idx in self.layer_cfgs, f"No AQ config found for layer {layer_idx}"
        cfg = self.layer_cfgs[layer_idx]

        N, num_heads, head_dim = keys.shape
        assert head_dim == cfg.d, f"head_dim={head_dim}, config.d={cfg.d}"
        assert num_heads % cfg.heads_per_group == 0, "num_heads must be divisible by heads_per_group"

        num_groups = num_heads // cfg.heads_per_group
        if max_head_groups is not None:
            num_groups = min(num_groups, max_head_groups)

        # Train one AQ codebook per head-group.
        for group_idx in range(num_groups):
            self.fit_layer_group(
                layer_idx=layer_idx,
                group_idx=group_idx,
                keys=keys,
                n_iter=n_iter,
                batch_size=batch_size,
                ema_decay=ema_decay,
                log_fn=log_fn,
                # Compatibility: when fit_layer drives only the first group, keep a short `layer{idx}: ` prefix.
                _use_layer_prefix_only=(max_head_groups == 1),
            )

    @torch.no_grad()
    def fit_layer_group(
        self,
        layer_idx: int,
        group_idx: int,
        keys: torch.Tensor,
        n_iter: int = 10,
        batch_size: int = 3_000_000,
        ema_decay: Optional[float] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        _use_layer_prefix_only: bool = False,
    ) -> None:
        """
        Train / incrementally update the AQ codebook for one (layer, head-group).

        This supports streaming training in the order:
          layer → group → segment → n_iter.
        """
        assert layer_idx in self.layer_cfgs, f"No AQ config found for layer {layer_idx}"
        cfg = self.layer_cfgs[layer_idx]

        N, num_heads, head_dim = keys.shape
        assert head_dim == cfg.d, f"head_dim={head_dim}, config.d={cfg.d}"
        assert num_heads % cfg.heads_per_group == 0, "num_heads must be divisible by heads_per_group"

        num_groups = num_heads // cfg.heads_per_group
        assert 0 <= group_idx < num_groups, f"group_idx={group_idx} out of range for num_groups={num_groups}"

        h_start = group_idx * cfg.heads_per_group
        h_end = h_start + cfg.heads_per_group

        # Extract keys for this head-group and flatten to [N * heads_per_group, d].
        group_keys = keys[:, h_start:h_end, :].reshape(-1, head_dim)
        group_keys = group_keys.to(device=self.device, dtype=self.dtype)

        # If this (layer, group) already has an AQ model, continue training; otherwise initialize a new one.
        aq = self._aq_per_group.get((layer_idx, group_idx))
        if aq is None:
            aq = AdditiveQuantizer(
                M=cfg.M,
                K=cfg.K,
                d=cfg.d,
                init="random",
                reg=1e-4,
                device=self.device,
                dtype=self.dtype,
            )

        # Set log prefix / external logger.
        if _use_layer_prefix_only:
            # Compatibility: only print a layer-level prefix.
            aq.log_prefix = f"layer{layer_idx}: "
        else:
            aq.log_prefix = f"layer{layer_idx}_group{group_idx}: "
        if log_fn is not None:
            aq.log_fn = log_fn

        aq.fit(
            group_keys,
            n_iter=n_iter,
            method="icm",
            update_method="batchwise",
            codebook_save_path=None,
            batch_size=batch_size,
            ema_decay=ema_decay,
        )
        # Precompute LUT after training.
        aq._precompute_icm_lut()
        self._aq_per_group[(layer_idx, group_idx)] = aq

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, save_dir: str) -> None:
        """
        Save all trained layer/group codebooks to disk.

        Format:
            {save_dir}/layer_{L}_group_{G}.pt:
                {
                    "M": int,
                    "K": int,
                    "d": int,
                    "codebooks": Tensor[M, K, d],
                }
        """
        os.makedirs(save_dir, exist_ok=True)
        for (layer_idx, group_idx), aq in self._aq_per_group.items():
            path = os.path.join(save_dir, f"layer_{layer_idx}_group_{group_idx}.pt")
            torch.save(
                {
                    "M": aq.M,
                    "K": aq.K,
                    "d": aq.d,
                    "codebooks": aq.codebooks.to("cpu"),
                },
                path,
            )

    def load(self, save_dir: str) -> None:
        """
        Load all layer/group codebooks from disk.

        Notes:
        - Filenames are expected to follow: layer_{L}_group_{G}.pt
        - Only files matching current `layer_cfgs` will be loaded; mismatched layers/groups are ignored.
        """
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(save_dir)

        for fname in os.listdir(save_dir):
            if not fname.startswith("layer_") or "_group_" not in fname:
                continue
            try:
                core, _ = os.path.splitext(fname)
                _, layer_str, _, group_str = core.split("_")
                layer_idx = int(layer_str)
                group_idx = int(group_str)
            except Exception:
                continue

            if layer_idx not in self.layer_cfgs:
                continue

            # If num_heads is available, skip stale group files that cannot be used by current config.
            cfg = self.layer_cfgs[layer_idx]
            if cfg.num_heads is not None and cfg.heads_per_group > 0:
                expected_groups = int(cfg.num_heads) // int(cfg.heads_per_group)
                if expected_groups > 0 and group_idx >= expected_groups:
                    # Likely leftover codebooks from older experiments (different heads_per_group).
                    continue

            path = os.path.join(save_dir, fname)
            ckpt = torch.load(path, map_location="cpu")
            M = ckpt["M"]
            K = ckpt["K"]
            d = ckpt["d"]
            codebooks = ckpt["codebooks"].to(device=self.device, dtype=self.dtype)

            # Keep runtime config consistent with checkpoint to avoid subtle bugs:
            # - K mismatch can cause uint8 codes compression overflow (K>256)
            # - d mismatch breaks attention head_dim assumptions
            # - M mismatch changes codes shape and LUT logic
            if (cfg.M != M) or (cfg.K != K) or (cfg.d != d):
                # Preserve grouping semantics; only update codebook shape metadata.
                self.layer_cfgs[layer_idx] = AQLayerConfig(
                    M=int(M),
                    K=int(K),
                    d=int(d),
                    heads_per_group=int(cfg.heads_per_group),
                    num_heads=cfg.num_heads,
                )

            aq = AdditiveQuantizer(
                M=M,
                K=K,
                d=d,
                init="pretrained",
                reg=1e-4,
                device=self.device,
                dtype=self.dtype,
                codebooks=codebooks,
            )
            aq._precompute_icm_lut()
            self._aq_per_group[(layer_idx, group_idx)] = aq

    # ------------------------------------------------------------------
    # Encoding and approximate inner-product scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_keys(
        self,
        layer_idx: int,
        head_idx: int,
        keys: torch.Tensor,
        batch_size: int = 3_000_000,
        icm_iters: int = 2,
    ) -> torch.Tensor:
        """
        Encode keys for one (layer, head) using the trained AQ codebook.

        Args:
        - layer_idx: layer index
        - head_idx: head index (mapped to a head-group internally)
        - keys: key vectors with shape [N, head_dim]

        Returns:
        - codes: [N, M] integer tensor; each column is the index of the corresponding sub-codebook entry.
        """
        cfg = self.layer_cfgs[layer_idx]
        group_idx = head_idx // cfg.heads_per_group
        aq = self._aq_per_group[(layer_idx, group_idx)]

        X = keys.to(device=self.device, dtype=self.dtype)
        if icm_iters == 0:
            fast_codes = _fast_greedy_encode(aq, X)
            if fast_codes is not None:
                return fast_codes
        # Prefill (icm_iters>0) is typically used in training/offline eval and may need loss.
        # Decode usually only needs codes; return_loss=False avoids extra reconstruction error work and sync.
        return_loss = icm_iters > 0
        codes, _, _ = aq.encode(
            X,
            method="icm",
            icm_iters=icm_iters,
            batch_size=batch_size,
            return_loss=return_loss,
        )

        return codes  # [N, M]

    @torch.no_grad()
    def encode_keys_batch(
        self,
        layer_idx: int,
        group_idx: int,
        keys: torch.Tensor,
        batch_size: int = 3_000_000,
        icm_iters: int = 2,
    ) -> torch.Tensor:
        """
        Batch-encode keys for one (layer, head-group).
        Useful when heads_per_group == num_heads (single group): you can pack all heads' keys into one batch
        to reduce Python overhead and kernel launches.

        Args:
        - layer_idx: layer index
        - group_idx: head-group index (0 when heads_per_group==num_heads)
        - keys: key vectors with shape [N_all, d]

        Returns:
        - codes: [N_all, M] integer tensor.
        """
        aq = self._aq_per_group[(layer_idx, group_idx)]

        X = keys.to(device=self.device, dtype=self.dtype)
        if icm_iters == 0:
            fast_codes = _fast_greedy_encode(aq, X)
            if fast_codes is not None:
                return fast_codes
        # Fallback path supports ICM refinement and CPU/non-CUDA execution.
        return_loss = icm_iters > 0
        codes, _, _ = aq.encode(
            X,
            method="icm",
            icm_iters=icm_iters,
            batch_size=batch_size,
            return_loss=return_loss,
        )

        return codes  # [N_all, M]

    @torch.no_grad()
    def reconstruct(
        self,
        layer_idx: int,
        head_idx: int,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct approximate key vectors from AQ codes.

        Args:
        - layer_idx/head_idx: selects the AQ codebook (by heads_per_group grouping)
        - codes: integer indices with shape [N, M]

        Returns:
        - recon: [N, d] reconstructed vectors (dtype follows the AQ manager)
        """
        cfg = self.layer_cfgs[layer_idx]
        group_idx = head_idx // cfg.heads_per_group
        aq = self._aq_per_group[(layer_idx, group_idx)]

        codes = codes.to(device=aq.codebooks.device, dtype=torch.long)
        N = codes.shape[0]
        recon = torch.zeros(N, cfg.d, device=aq.codebooks.device, dtype=aq.codebooks.dtype)
        for m in range(aq.M):
            recon += aq.codebooks[m][codes[:, m]]
        return recon

    @torch.no_grad()
    def approx_inner_product(
        self,
        layer_idx: int,
        head_idx: int,
        q: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate inner products between a query q and a set of encoded keys using AQ LUT.

        Args:
        - layer_idx: layer index
        - head_idx: head index (mapped to a head-group)
        - q: query vector, [d] or [1, d]
        - codes: AQ codes, [N, M]

        Returns:
        - scores: [N] approximate inner products (for ranking / top-k selection).

        Math:
          k_i ≈ sum_m C_m[z_{i,m}]
          s_i ≈ sum_m (q · C_m[z_{i,m}])
        where q · C_m[k] is precomputed as a LUT and then gathered/summed.
        """
        cfg = self.layer_cfgs[layer_idx]
        group_idx = head_idx // cfg.heads_per_group
        aq = self._aq_per_group[(layer_idx, group_idx)]

        # Normalize q to [d].
        if q.dim() == 2:
            assert q.shape[0] == 1
            q_vec = q[0]
        else:
            q_vec = q
        q_vec = q_vec.to(device=self.device, dtype=self.dtype)  # [d]

        # LUT: for each sub-codebook m, precompute q · C_m[k] with shape [M, K].
        M, K, d = aq.codebooks.shape
        assert M == aq.M and K == aq.K and d == aq.d

        # [M, K, d] · [d] -> [M, K]
        lut = torch.einsum("mkd,d->mk", aq.codebooks, q_vec)  # (M, K)

        # Gather from LUT using codes and sum across m.
        # Note: torch.gather indices must be int64/long. If upstream stores codes as uint8/int8,
        # we cast here (this only happens on the PyTorch path; decode hot path should use CUDA kernels).
        if codes.dtype != torch.long:
            codes = codes.to(device=self.device, dtype=torch.long)
        else:
            codes = codes.to(device=self.device)
        N, M_ = codes.shape
        assert M_ == M

        # Vectorized gather: transpose codes -> [M, N] and gather along dim=1.
        codes_T = codes.t()  # [M, N]
        # gathered[m, n] = lut[m, codes_T[m, n]]
        gathered = lut.gather(dim=1, index=codes_T)  # [M, N]

        scores = gathered.sum(dim=0)  # [N]

        return scores

    @torch.no_grad()
    def approx_inner_product_multihead(
        self,
        layer_idx: int,
        q: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized version: compute AQ LUT inner-product scores for many heads in one layer.

        Args:
        - layer_idx: layer index
        - q: [Q, d] queries (typically Q = B * H)
        - codes: [Q, N, M] AQ codes

        Returns:
        - scores: [Q, N] approximate inner products
        """
        cfg = self.layer_cfgs[layer_idx]
        # Assumption: all heads in this layer share one codebook (heads_per_group covers all heads),
        # matching heads_per_group=0 (auto-share) used by training/eval configs.
        assert cfg.heads_per_group >= 1, "approx_inner_product_multihead requires valid heads_per_group"
        group_idx = 0
        aq = self._aq_per_group[(layer_idx, group_idx)]

        # q: [Q, d]
        if q.dim() == 1:
            q = q[None, :]
        q = q.to(device=self.device, dtype=self.dtype)  # [Q, d]

        # codes: [Q, N, M]
        # Note: upstream may store codes as uint8/int8 to save memory.
        # - CUDA kernels can read uint8/int8 directly (no cast);
        # - PyTorch fallback gather needs int64 indices, so we cast only on fallback.
        codes = codes.to(device=self.device)
        Q, N, M = codes.shape

        M_cb, K, d = aq.codebooks.shape
        assert M_cb == aq.M and M_cb == M and K == aq.K and d == aq.d

        # Prefer CUDA kernels (when available on CUDA); otherwise fall back to PyTorch.
        # For small-batch profiling, you can force-disable CUDA kernels via USE_AQ_LUT_CUDA.
        # CUDA LUT kernel supports fp32 and fp16 (fp16 path uses fp32 accumulate).
        # For other dtypes, fallback to PyTorch path.
        if (
            USE_AQ_LUT_CUDA
            and HAS_AQ_LUT_CUDA
            and q.is_cuda
            and aq.codebooks.is_cuda
            and codes.is_cuda
            and q.dtype == aq.codebooks.dtype
            and q.dtype in (torch.float16, torch.float32)
        ):
            # Fast path (recommended): GEMM builds lut[q, m*K+k], then a lightweight gather+sum kernel
            # reads uint8/int8 codes directly (avoids int64 cast and per-n dot(d)).
            if aq_lut_gather_cuda is not None:
                # codebooks_flat: [M*K, d]
                cb = aq.codebooks
                M_cb, K, d = cb.shape
                cb_flat = cb.reshape(M_cb * K, d)
                # lut: [Q, M*K]
                # NOTE: matmul is tiny here (Q ~ B*H, d ~ 128, M*K ~ 2048) and is usually much faster than the old kernel.
                lut = torch.matmul(q, cb_flat.t())
                # ensure contiguous for extension
                if not codes.is_contiguous():
                    codes = codes.contiguous()
                if not lut.is_contiguous():
                    lut = lut.contiguous()
                scores = aq_lut_gather_cuda(lut, codes)  # [Q, N]
                return scores

            # Fallback to the older CUDA kernel (per-n dot(d)); still supports uint8/int8 but usually slower.
            if not codes.is_contiguous():
                codes = codes.contiguous()
            scores = aq_lut_cuda(q, aq.codebooks, codes)  # [Q, N]
            return scores

        # -----------------------------
        # PyTorch fallback (kept consistent with the single-head implementation).
        # -----------------------------
        # IMPORTANT:
        # For bf16 models (e.g., Qwen2.5) we often fall back here because CUDA LUT kernels
        # only support fp16/fp32. If we cast the full `codes` to int64 at once, memory can
        # explode when Q is large (prefill_q_chunk * num_heads).
        #
        # To avoid OOM, bound both dimensions.  Decode has a small Q (normally the
        # number of query heads) but a very large N; casting all [Q,N,M] uint8 codes
        # to int64 at 128K still costs about 1 GiB.  Token chunking keeps the temporary
        # index and gather tensors small even when the optional CUDA extension is absent.
        # Short-context PPL evaluates many query rows at once and has ample room
        # for a larger block; long-context decode has tiny Q but a very large N.
        # Choose the block from N so 8K evaluation remains fast without bringing
        # back the 128K decode memory spike.
        q_chunk = 512 if N <= 8192 else 64
        token_chunk = 8192
        scores_out = torch.empty((Q, N), device=q.device, dtype=q.dtype)
        for start in range(0, Q, q_chunk):
            end = min(start + q_chunk, Q)
            q_sub = q[start:end]  # [qC, d]
            lut_sub = torch.einsum("mkd,qd->qmk", aq.codebooks, q_sub)  # [qC, M, K]
            for token_start in range(0, N, token_chunk):
                token_end = min(token_start + token_chunk, N)
                codes_sub = codes[start:end, token_start:token_end, :]
                if codes_sub.dtype != torch.long:
                    codes_sub = codes_sub.to(dtype=torch.long)
                codes_T = codes_sub.permute(0, 2, 1)  # [qC, M, nC]
                gathered = lut_sub.gather(dim=2, index=codes_T)  # [qC, M, nC]
                scores_out[start:end, token_start:token_end] = gathered.sum(dim=1)

        return scores_out

    @torch.no_grad()
    def approx_inner_product_multihead_gqa(
        self,
        layer_idx: int,
        q: torch.Tensor,
        codes: torch.Tensor,
        num_kv_group: int,
        scoring_m: int = 32,
    ) -> torch.Tensor:
        """GQA-native AQ scoring without expanding KV-head codes to query heads.

        Args:
        - q: [B*Hq, d]
        - codes: [B*Hkv, N, M]
        - num_kv_group: Hq // Hkv
        """
        if q.dim() == 1:
            q = q[None, :]
        q = q.to(device=self.device, dtype=self.dtype)
        codes = codes.to(device=self.device)
        q_heads, _ = q.shape
        kv_heads, n_tokens, m_codes = codes.shape
        group = int(num_kv_group)
        if q_heads != kv_heads * group:
            raise ValueError(
                f"Invalid GQA score heads: Hq={q_heads}, Hkv={kv_heads}, G={group}"
            )

        aq = self._aq_per_group[(layer_idx, 0)]
        m_cb, k_cb, d_cb = aq.codebooks.shape
        scoring_m = int(scoring_m)
        if scoring_m != 32:
            raise ValueError(f"aq_scoring_m must be 32, got {scoring_m}")
        if m_cb != m_codes or d_cb != q.shape[-1]:
            raise ValueError(
                f"AQ shape mismatch: codebook={tuple(aq.codebooks.shape)}, "
                f"q={tuple(q.shape)}, codes={tuple(codes.shape)}"
            )

        if (
            USE_AQ_LUT_CUDA
            and HAS_AQ_LUT_GQA_CUDA
            and q.is_cuda
            and aq.codebooks.is_cuda
            and codes.is_cuda
            and q.dtype == torch.float16
            and aq.codebooks.dtype == torch.float16
            and codes.dtype == torch.uint8
        ):
            lut = torch.matmul(q, aq.codebooks.reshape(m_cb * k_cb, d_cb).t())
            lut = lut.reshape(q_heads, m_cb, k_cb).contiguous()
            if aq_lut_gather_grouped is not None and group <= 4:
                return aq_lut_gather_grouped(lut, codes, group)
            return aq_lut_gather_gqa(
                lut, codes, group, 0, m_cb, None
            )

        if scoring_m != 32:
            raise RuntimeError(
                "Only full-codebook AQ scoring is supported; "
                "refusing to silently fall back to exact M=32 scoring"
            )

        # Correct fallback for environments where the GQA kernel is unavailable.
        expanded = codes.reshape(kv_heads, 1, n_tokens, m_codes).expand(
            kv_heads, group, n_tokens, m_codes
        ).reshape(q_heads, n_tokens, m_codes)
        return self.approx_inner_product_multihead(layer_idx, q, expanded)


class ShardedAQLayerCodebookManager:
    """
    A thin wrapper that keeps one AQLayerCodebookManager per device.

    Motivation:
    - When the model is sharded across GPUs (device_map="auto"), a single manager on cuda:0
      forces cross-device copies for layer attention on other GPUs, inflating aq_post.
    - Codebooks are read-only and typically small enough to replicate per GPU.

    Behavior:
    - A "default" manager is created eagerly on the first device.
    - Other devices are created lazily on first use.
    - encode/approx APIs dispatch based on the input tensor's device.
    """

    def __init__(
        self,
        layer_cfgs: Dict[int, AQLayerConfig],
        aq_codebook_dir: str,
        devices: Optional[list] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.layer_cfgs = layer_cfgs
        self.aq_codebook_dir = aq_codebook_dir
        self.dtype = dtype

        if devices is None or len(devices) == 0:
            devices = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]

        self.devices = sorted({torch.device(d) for d in devices}, key=lambda d: str(d))
        self._mgr_by_device: Dict[str, AQLayerCodebookManager] = {}

        self._default_device = self.devices[0]
        self._default_mgr = self._get_mgr(self._default_device)
        self._aq_per_group = self._default_mgr._aq_per_group

    def _dev_key(self, device: torch.device) -> str:
        d = torch.device(device)
        return str(d)

    def _get_mgr(self, device: torch.device) -> AQLayerCodebookManager:
        key = self._dev_key(device)
        mgr = self._mgr_by_device.get(key)
        if mgr is not None:
            return mgr
        mgr = AQLayerCodebookManager(self.layer_cfgs, device=torch.device(device), dtype=self.dtype)
        mgr.load(self.aq_codebook_dir)
        self._mgr_by_device[key] = mgr
        return mgr

    def _pick_device_from_tensor(self, t: torch.Tensor) -> torch.device:
        return t.device if isinstance(t, torch.Tensor) else self._default_device

    @torch.no_grad()
    def encode_keys(self, layer_idx: int, head_idx: int, keys: torch.Tensor, **kwargs) -> torch.Tensor:
        mgr = self._get_mgr(self._pick_device_from_tensor(keys))
        return mgr.encode_keys(layer_idx, head_idx, keys, **kwargs)

    @torch.no_grad()
    def encode_keys_batch(self, layer_idx: int, group_idx: int, keys: torch.Tensor, **kwargs) -> torch.Tensor:
        mgr = self._get_mgr(self._pick_device_from_tensor(keys))
        return mgr.encode_keys_batch(layer_idx, group_idx, keys, **kwargs)

    @torch.no_grad()
    def reconstruct(self, layer_idx: int, head_idx: int, codes: torch.Tensor) -> torch.Tensor:
        mgr = self._get_mgr(self._pick_device_from_tensor(codes))
        return mgr.reconstruct(layer_idx, head_idx, codes)

    @torch.no_grad()
    def approx_inner_product(
        self,
        layer_idx: int,
        head_idx: int,
        q: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        mgr = self._get_mgr(self._pick_device_from_tensor(q))
        return mgr.approx_inner_product(layer_idx, head_idx, q, codes)

    @torch.no_grad()
    def approx_inner_product_multihead(
        self,
        layer_idx: int,
        q: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        mgr = self._get_mgr(self._pick_device_from_tensor(q))
        return mgr.approx_inner_product_multihead(layer_idx, q, codes)

    @torch.no_grad()
    def approx_inner_product_multihead_gqa(
        self,
        layer_idx: int,
        q: torch.Tensor,
        codes: torch.Tensor,
        num_kv_group: int,
        scoring_m: int = 32,
    ) -> torch.Tensor:
        mgr = self._get_mgr(self._pick_device_from_tensor(q))
        return mgr.approx_inner_product_multihead_gqa(
            layer_idx, q, codes, num_kv_group, scoring_m=scoring_m
        )

@torch.no_grad()
def aq_iou_topk(
    manager: AQLayerCodebookManager,
    layer_idx: int,
    head_idx: int,
    q: torch.Tensor,
    keys: torch.Tensor,
    top_k: int,
    codes: Optional[torch.Tensor] = None,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    IoU evaluation for one (layer, head) with one query using AQ approximate inner products.

    Args:
    - manager: trained AQLayerCodebookManager
    - layer_idx / head_idx: which layer/head to evaluate
    - q: query vector, [d] or [1, d] (typically the last-token head representation)
    - keys: key vectors, [N, d] (all tokens for this layer/head)
    - top_k: how many keys to keep for IoU
    - codes: optional precomputed codes, [N, M]; if None, encode_keys() will be called

    Returns:
    - iou: IoU between approx top-k and exact top-k (0..1)
    - approx_indices: top-k indices from AQ approximation
    - true_indices: top-k indices from exact inner products
    """
    # 1) Exact inner products.
    if q.dim() == 2:
        assert q.shape[0] == 1
        q_vec = q[0]
    else:
        q_vec = q
    q_vec = q_vec.to(device=keys.device, dtype=keys.dtype)  # [d]

    true_scores = torch.matmul(keys, q_vec)  # [N]

    # 2) If codes are not provided, encode keys via the AQ manager.
    if codes is None:
        codes = manager.encode_keys(layer_idx, head_idx, keys)

    # 3) AQ approximate inner products.
    approx_scores = manager.approx_inner_product(layer_idx, head_idx, q_vec, codes)

    # 4) Take top-k and compute IoU.
    top_k = min(top_k, keys.shape[0])
    _, approx_idx = torch.topk(approx_scores, k=top_k, dim=0)
    _, true_idx = torch.topk(true_scores, k=top_k, dim=0)

    approx_set = set(approx_idx.tolist())
    true_set = set(true_idx.tolist())

    intersect = approx_set.intersection(true_set)
    union = approx_set.union(true_set)
    iou = len(intersect) / len(union) if len(union) > 0 else 0.0

    return float(iou), approx_idx, true_idx
