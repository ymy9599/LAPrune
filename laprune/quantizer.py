import numpy as np
import torch
import torch.nn as nn


def gen_nums_list(min_value, nums, N):
    # y ~ Dirichlet 保证非负且和为 1
    y = np.random.dirichlet(alpha=np.ones(nums))  # 可把 alpha 全设为>1 更接近平均
    x = (min_value + (1 - nums * min_value) * y) * N  # 每个 ≥min_value，总和=1
    nums_list = []

    x_start = 0
    for i in range(nums):
        nums_list.append([x_start, int(x_start + x[i])])
        x_start = int(x_start + x[i])
    nums_list[-1][1] = N
    return nums_list


def quantize_linear(X, B=8):
    eps = 1e-6
    X_min = X.min(dim=1, keepdim=True).values
    X_max = X.max(dim=1, keepdim=True).values
    S = (X_max - X_min) / (2**B - 1)
    S = S.clamp(min=eps)
    Q = torch.floor((X - X_min) / S)
    X_rec = Q * S + X_min
    return Q, X_rec, S, X_min


@torch.no_grad()
def _sym_invsqrt_from_cov(S: torch.Tensor, eps: float = 1e-6):
    """S≈Σ，返回 Σ^{-1/2} 和 Σ^{+1/2}（数值稳健的对称根）"""
    w, V = torch.linalg.eigh((S + S.T) * 0.5)
    w = torch.clamp(w, min=eps)
    Winv = (V * (1.0 / torch.sqrt(w))).matmul(V.T)  # Σ^{-1/2}
    Wpos = (V * torch.sqrt(w)).matmul(V.T)  # Σ^{+1/2}
    return Winv, Wpos


@torch.no_grad()
def _kmeanspp_seeds(X: torch.Tensor, K: int) -> torch.Tensor:
    """KMeans++ 选 K 个初始中心。X: (n,d) fp32 on device"""
    n, d = X.shape
    inds = torch.empty(K, dtype=torch.long, device=X.device)
    inds[0] = torch.randint(0, n, (1,), device=X.device)
    centers = X[inds[0]].unsqueeze(0)  # (1,d)
    d2 = ((X - centers[0]) ** 2).sum(-1)  # (n,)
    for k in range(1, K):
        d2 = torch.clamp(d2, min=0)
        s = d2.sum()
        if s <= 0:
            inds[k] = torch.randint(0, n, (1,), device=X.device)
        else:
            inds[k] = torch.multinomial(d2 / s, 1).item()
        c = X[inds[k]]
        centers = torch.cat([centers, c.unsqueeze(0)], 0)
        d2 = torch.minimum(d2, ((X - c) ** 2).sum(-1))
    return centers  # (K,d)


@torch.no_grad()
def _minibatch_kmeans(
    X: torch.Tensor, K: int, iters: int = 6, batch: int = 531072
) -> torch.Tensor:
    """
    简洁稳健的 Mini-Batch Lloyd：KMeans++ seeds + EMA 更新。
    X: (n,d) fp32
    返回 (K,d)
    """
    device = X.device
    n, d = X.shape
    C = _kmeanspp_seeds(X, K)  # (K,d)
    usage = torch.ones(K, device=device)
    beta = 0.9  # EMA 衰减
    B = min(batch, n)

    for _ in range(iters):
        idx = torch.randint(0, n, (B,), device=device)
        xb = X[idx]  # (B,d)
        rr = (xb * xb).sum(1, keepdim=True)  # (B,1)
        cc = (C * C).sum(1).unsqueeze(0)  # (1,K)
        rc = xb @ C.T  # (B,K)
        d2 = rr + cc - 2 * rc
        assign = d2.argmin(dim=1)  # (B,)

        sums = torch.zeros_like(C)
        cnts = torch.zeros(K, device=device)
        sums.index_add_(0, assign, xb)
        cnts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.float32))
        hit = cnts > 0
        C[hit] = beta * C[hit] + (1 - beta) * (sums[hit] / cnts[hit].unsqueeze(1))
        usage = beta * usage + (1 - beta) * cnts

    # 温和范数归一，抑制极端范数
    rms = C.pow(2).mean(1).sqrt()
    tgt = rms.median()
    scale = (tgt / (rms + 1e-8)).clamp(0.5, 2.0).unsqueeze(1)
    C = C * scale
    return C


@torch.no_grad()
def init_laprune_codebooks_stable_mkd(
    X: torch.Tensor,
    M: int,
    K: int,
    sample_size: int = 200000,
    kmeans_iters: int = 6,
    out_dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    生成稳定的 LAPrune 码本：返回 (M, K, d) 的 Tensor。
    步骤：抽样 -> 白化 -> 逐级残差 MiniBatch KMeans++ -> 映回原空间并均摊均值。
    - 只在初始化/训练期使用白化；最终返回已在原空间的码本，不需要保存 W/μ。
    """
    device = device or X.device
    out_dtype = out_dtype or X.dtype
    N, d = X.shape

    # 1) 大样本子采样并转 fp32
    if sample_size < N:
        idx = torch.randint(0, N, (sample_size,), device=device)
        Xs = X[idx].to(device=device, dtype=torch.float32)
    else:
        Xs = X.to(device=device, dtype=torch.float32)

    # 2) 估计 μ, Σ 并白化
    mu = Xs.mean(0)  # (d,)
    Xc = Xs - mu
    Sigma = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    W, Winv = _sym_invsqrt_from_cov(Sigma, eps=1e-6)  # W=Σ^{-1/2}, Winv=Σ^{+1/2}
    Y = Xc @ W.T  # (n,d), 近似各向同性

    # 3) 逐级残差：依次拟合每个子码本
    codebooks_w = torch.empty((M, K, d), device=device, dtype=torch.float32)
    R = Y.clone()
    for m in range(M):
        Cw = _minibatch_kmeans(R, K, iters=kmeans_iters)  # (K,d)
        codebooks_w[m] = Cw
        rr = (R * R).sum(1, keepdim=True)  # (n,1)
        cc = (Cw * Cw).sum(1).unsqueeze(0)  # (1,K)
        rc = R @ Cw.T  # (n,K)
        assign = (rr + cc - 2 * rc).argmin(1)  # (n,)
        R = R - Cw[assign]

    # 4) 映回原空间并均摊均值（避免额外保存 μ）
    codebooks = torch.empty((M, K, d), device=device, dtype=out_dtype)
    bias_share = (mu / M).to(dtype=out_dtype)
    for m in range(M):
        Cm = codebooks_w[m] @ Winv.T  # (K,d) fp32
        Cm = Cm.to(out_dtype) + bias_share
        codebooks[m] = Cm

    return codebooks


def compute_reconstruction_error(X, X_rec):
    return torch.sqrt(((X - X_rec) ** 2).sum(1)).sum()


class AdditiveQuantizer:
    def __init__(
        self,
        M,
        K,
        d,
        init="random",
        reg=1e-4,
        device=None,
        dtype=torch.float32,
        codebooks=None,
    ):
        self.M = M
        self.K = K
        self.d = d
        self.reg = reg
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # 日志相关：可选的前缀与外部日志函数（例如写入文件）
        self.log_prefix: str = ""
        self.log_fn = None
        # Initialize codebooks
        if init == "random":
            self.codebooks = None
        else:
            self.codebooks = codebooks
            self.to(device=device, dtype=dtype)
        # containers for ICM LUT
        self.codebook_norms = None
        self.codebooks_T = None
        self.total_iter = 0

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
            self.codebooks = self.codebooks.to(device)
        if dtype is not None:
            self.dtype = dtype
            self.codebooks = self.codebooks.to(dtype=dtype)
        return self

    def _convert_input(self, X):
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        return X.to(device=self.device, dtype=self.dtype)

    def fit(
        self,
        X,
        n_iter=10,
        method="icm",
        update_method="batchwise",
        codebook_save_path=None,
        batch_size=3000000,
        ema_decay=None,
    ):
        X = self._convert_input(X)
        N = X.shape[0]
        if self.codebooks is None:
            self.codebooks = init_laprune_codebooks_stable_mkd(
                X,
                self.M,
                self.K,
                sample_size=5000000,
                kmeans_iters=6,
                device=self.device,
                out_dtype=self.dtype,
            )
        self._precompute_icm_lut()

        loss_min = float("inf")
        total_mean_rel_loss_min = float("inf")
        iter_num = 0
        X_orig = X
        for it in range(n_iter):
            indices = torch.randperm(N, device=self.device)
            nums = gen_nums_list(0.4, 2, N)
            total_mean_rel_loss1 = 0.0
            loss_cur1 = 0.0
            total_num = 0
            for s_e in nums:
                X_batch = X_orig[indices[s_e[0] : s_e[1]]].clone()
                total_num += X_batch.shape[0]
                codes, loss, _ = self.encode(X_batch, method=method, batch_size=batch_size)
                if update_method == "batchwise":
                    self._update_codebooks_batchwise(X_batch, codes, batch_size=batch_size, ema_decay=ema_decay)
                else:
                    raise ValueError(f"Unknown update_method: {update_method}")

                loss_our, loss_rel_our = self._compute_augmented_loss(X_batch, codes)
                loss_cur1 = loss_our + loss_cur1
                total_mean_rel_loss1 = loss_rel_our + total_mean_rel_loss1
            loss_cur1 = loss_cur1 / total_num
            total_mean_rel_loss1 = total_mean_rel_loss1 / total_num

            if it == n_iter - 1:
                msg = f"{self.log_prefix}reconstruction loss {loss_cur1} reconstruction rel loss {total_mean_rel_loss1}"
                if self.log_fn is not None:
                    self.log_fn(msg)
                else:
                    print(msg)

            self.total_iter += 1
            if loss_cur1 < loss_min:
                loss_min = loss_cur1
                total_mean_rel_loss_min = total_mean_rel_loss1
                iter_num = 0
                self.best_codebooks = self.codebooks.clone()
                if codebook_save_path:
                    torch.save(self.best_codebooks, codebook_save_path)
            elif loss_cur1 == loss_min:
                iter_num += 1
                if total_mean_rel_loss_min > total_mean_rel_loss1:
                    total_mean_rel_loss_min = total_mean_rel_loss1
                    self.best_codebooks = self.codebooks.clone()
                    if codebook_save_path:
                        torch.save(self.best_codebooks, codebook_save_path)
            else:
                iter_num += 1
            if iter_num > 7:
                self.codebooks = self.best_codebooks.clone()
                self._precompute_icm_lut()
                break
        self._precompute_icm_lut()
        return self

    def _compute_augmented_loss(self, X, codes, batch_size=5000000):
        N, d = X.shape
        loss_sum = 0.0
        loss_rel_sum = 0.0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb = X[start:end]
            cb = codes[start:end]

            recon_b = torch.zeros_like(xb)
            for m in range(self.M):
                recon_b += self.codebooks[m][cb[:, m]]

            abs_error = xb - recon_b
            abs_loss = torch.sqrt((abs_error**2).sum(1)).sum()

            rel_error = abs_error / xb
            rel_error = torch.nan_to_num(rel_error, nan=1.0, posinf=1.0, neginf=1.0)
            rel_loss = torch.sqrt((rel_error**2).sum(1)).sum()

            loss_sum += abs_loss.item()
            loss_rel_sum += rel_loss.item()

        return loss_sum, loss_rel_sum

    def _precompute_icm_lut(self):
        self.codebook_norms = [torch.sum(cb**2, dim=1) for cb in self.codebooks]
        self.codebooks_T = [cb.T.clone() for cb in self.codebooks]

    def encode(self, X, method="icm", icm_iters=2, batch_size=3000000, return_loss: bool = True):
        """
        使用 ICM+LUT 对输入向量进行加性量化编码。
        """
        X = self._convert_input(X)
        N = X.shape[0]
        if method == "icm":
            codes = torch.zeros(N, self.M, dtype=torch.long, device=self.device)
            if return_loss:
                total_loss = 0.0
                for i in range(0, N, batch_size):
                    end = min(i + batch_size, N)
                    batch = X[i:end]
                    c, loss, _ = self._batch_icm_lut(batch, icm_iters, return_loss=True)
                    codes[i:end] = c
                    total_loss += loss * (end - i)
                avg_loss = total_loss / N
            else:
                for i in range(0, N, batch_size):
                    end = min(i + batch_size, N)
                    batch = X[i:end]
                    c = self._batch_icm_lut(batch, icm_iters, return_loss=False)
                    codes[i:end] = c
                avg_loss = 0.0
            return codes, avg_loss, None
        raise NotImplementedError("Only ICM method is implemented in this LUT version.")

    def _batch_icm_lut(self, X, iters, return_loss=True):
        B, d = X.shape
        device = X.device
        codes = torch.zeros(B, self.M, dtype=torch.long, device=device)
        residual = X.clone()
        for m in range(self.M):
            r_norm = torch.sum(residual**2, dim=1, keepdim=True)
            dists = r_norm + self.codebook_norms[m].unsqueeze(0) - 2 * (residual @ self.codebooks_T[m])
            codes[:, m] = torch.argmin(dists, dim=1)
            residual -= self.codebooks[m][codes[:, m]]
        full_recon = X - residual
        for _ in range(iters):
            total_changed = 0
            for m in range(self.M):
                prev = codes[:, m]
                recon_wo = full_recon - self.codebooks[m][prev]
                residual = X - recon_wo
                r_norm = torch.sum(residual**2, dim=1, keepdim=True)
                dists = r_norm + self.codebook_norms[m].unsqueeze(0) - 2 * (residual @ self.codebooks_T[m])
                new = torch.argmin(dists, dim=1)
                total_changed += (new != prev).sum().item()
                codes[:, m] = new
                full_recon = recon_wo + self.codebooks[m][new]
            if total_changed == 0:
                break
        if return_loss:
            loss = torch.sqrt(((X - full_recon) ** 2).sum(dim=1)).sum() / B
            return codes, loss, None
        return codes

    @torch.no_grad()
    def _update_codebooks_batchwise(self, X, codes, batch_size=3000000, ema_decay=None, dtype_acc=torch.float32):
        """
        固定 codes 的闭式 M 步：
        对每个子码本 m、每个码字 k：
            C_m[k] ← 该码字下“排除 m 的残差”的均值
        """
        N, d = X.shape
        M, K = self.M, self.K
        device = self.device

        for m in range(M):
            code_sum = torch.zeros(K, d, device=device, dtype=dtype_acc)
            code_count = torch.zeros(K, device=device, dtype=torch.int64)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                xb = X[start:end].to(device)
                idxb = codes[start:end].to(device)
                B = xb.shape[0]

                xb_acc = xb.to(dtype_acc)
                recon_all = torch.zeros(B, d, device=device, dtype=dtype_acc)
                for j in range(M):
                    recon_all.add_(self.codebooks[j][idxb[:, j]].to(dtype_acc))

                contrib_m = self.codebooks[m][idxb[:, m]].to(dtype_acc)
                res_m = xb_acc - (recon_all - contrib_m)

                code_sum.index_add_(0, idxb[:, m], res_m)
                code_count.index_add_(0, idxb[:, m], torch.ones(B, device=device, dtype=torch.int64))

            cnt_f = code_count.to(dtype_acc)
            cnt_clamped = torch.clamp_min(cnt_f, 1.0)
            new_mean = (code_sum / cnt_clamped.unsqueeze(1)).to(self.codebooks[m].dtype)
            self.codebooks[m].copy_(new_mean)

        self._precompute_icm_lut()


__all__ = [
    "AdditiveQuantizer",
    "init_laprune_codebooks_stable_mkd",
    "quantize_linear",
    "compute_reconstruction_error",
]


