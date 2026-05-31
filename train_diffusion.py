#!/usr/bin/env python3
"""
DDPM on radio cluster maps — unconditional or scalar-conditional (TSC).

Unconditional mode (default) was used for the v1/v2 feasibility runs:
  - decisive memorization test (mean-subtracted NN distance + brightness guard)
  - normalized-space PSD as the meaningful fidelity readout

Conditional mode (--condition) adds TSC-scalar conditioning for synthetic-
data augmentation. Implementation:
  - Fourier-embed the (normalized) TSC scalar, MLP to time-embedding dim,
    add to the timestep embedding before the ResBlocks see it.
  - Learned **null token** in place of the condition embedding when the
    condition is NaN; classifier-free guidance dropout at training time
    (--cond-drop-prob, default 0.1) and CFG scale at sampling time
    (--cfg-scale).
  - At the end of training, sample at a grid of TSC values
    (--sample-tsc) and report whether morphology actually shifts with the
    condition (the gate for augmentation being useful).

Cluster-level split (3 projections per halo kept in the same fold) reuses
the project's no-leakage rule. Outputs go to --out-dir.

Usage:
  # unconditional (the v2 baseline)
  python train_diffusion.py --data diffusion_radio_64_v2.h5 --epochs 400

  # conditional on merger-TSC (Lee-style label, 0–7.7 Gyr)
  python train_diffusion.py --data diffusion_radio_64_v2.h5 --epochs 400 \\
      --condition --labels TSC_Cutimages/TSC_eachhalo_snap99.hdf5 \\
      --label-key tsc_gyr --cond-scale-norm 8.0 \\
      --sample-tsc 0.5 1.5 3.0 5.0 7.0 --n-per-tsc 16 --cfg-scale 1.5
"""

import argparse
import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

OUT = "diffusion_out"  # overridden by --out-dir at runtime


# ----------------------------- data ---------------------------------------
class RadioMaps(Dataset):
    """Projection-level grayscale maps; 8x rot/flip aug (physically valid).
    Optionally returns a per-sample scalar label for conditional training."""

    def __init__(self, imgs, labels=None, train=True, seed=0):
        self.imgs = imgs                              # (M, 1, S, S) in [-1, 1]
        self.labels = labels                          # (M,) float or None
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.imgs[i]
        if self.train:
            k = self.rng.integers(4)
            x = np.rot90(x, k, axes=(1, 2))
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=2)
            x = np.ascontiguousarray(x)
        x_t = torch.from_numpy(x)
        if self.labels is None:
            return x_t
        return x_t, torch.tensor(self.labels[i], dtype=torch.float32)


def load_split(path, val_frac=0.15, seed=0,
               labels_path=None, label_key="tsc_gyr", cond_scale_norm=1.0):
    """Returns (tr_imgs, tr_labels, val_imgs, val_labels, attrs).
    *_labels is None when labels_path is None (unconditional). Labels are
    normalised by cond_scale_norm so the model sees a roughly [0, 1]-scaled
    scalar (the same Fourier embedding spec as the timestep)."""
    with h5py.File(path, "r") as f:
        imgs = f["images"][:]                       # (N, 3, S, S)
        halo = f["meta/halo_id"][:]
        attrs = dict(f.attrs)

    labels = None
    if labels_path is not None:
        with h5py.File(labels_path, "r") as f:
            lhid = f["halo_id"][:]
            lval = f[label_key][:]
        lmap = {int(h): float(v) for h, v in zip(lhid, lval)}
        labels = np.array([lmap.get(int(h), np.nan) for h in halo],
                          dtype=np.float32) / cond_scale_norm

    rng = np.random.default_rng(seed)
    uniq = np.unique(halo)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac)
    val_h = set(uniq[:n_val].tolist())
    tr_m = np.array([h not in val_h for h in halo])

    def expand(mask):
        sel = imgs[mask]                            # (n, 3, S, S)
        out_imgs = sel.reshape(-1, 1, *sel.shape[2:]).astype(np.float32)
        if labels is None:
            return out_imgs, None
        out_labels = np.repeat(labels[mask], 3)     # (n*3,)
        return out_imgs, out_labels

    tr_i, tr_l = expand(tr_m)
    val_i, val_l = expand(~tr_m)
    return tr_i, tr_l, val_i, val_l, attrs


# --------------------------- model ----------------------------------------
def timestep_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class CondGroupNorm(nn.Module):
    """GroupNorm with per-channel scale/shift produced from an external
    embedding. Zero-initialised → identity at start so training begins
    from the same effective state as plain GroupNorm."""

    def __init__(self, groups, channels, emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, affine=False)
        self.proj = nn.Linear(emb_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, emb):
        h = self.norm(x)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    """If ada=True, the two GroupNorms become CondGroupNorms driven by the
    (time + condition) embedding; the separate additive time projection is
    dropped since AdaGN already carries that signal. Standard Imagen/EDM2
    block layout for scalar conditioning."""

    def __init__(self, cin, cout, temb, ada=False):
        super().__init__()
        self.ada = ada
        if ada:
            self.n1 = CondGroupNorm(8, cin, temb)
            self.n2 = CondGroupNorm(8, cout, temb)
        else:
            self.n1 = nn.GroupNorm(8, cin)
            self.n2 = nn.GroupNorm(8, cout)
            self.temb = nn.Linear(temb, cout)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        if self.ada:
            h = self.c1(F.silu(self.n1(x, t)))
            h = self.c2(F.silu(self.n2(h, t)))
        else:
            h = self.c1(F.silu(self.n1(x)))
            h = h + self.temb(t)[:, :, None, None]
            h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        a = torch.softmax(q @ k / math.sqrt(C), dim=-1)
        o = (a @ v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(o)


class UNet(nn.Module):
    """Symmetric UNet: per level a ResBlock (skip saved at that resolution),
    then strided downsample; mirror on the way up with transpose-conv.
    Optional scalar conditioning via Fourier-embed + MLP, summed into the
    timestep embedding; a learned null token replaces the embedding when
    the condition is NaN (for classifier-free guidance)."""

    def __init__(self, ch=64, mults=(1, 2, 4), cond=False, ada=False):
        super().__init__()
        temb = ch * 4
        self.tproj = nn.Sequential(nn.Linear(ch, temb), nn.SiLU(),
                                   nn.Linear(temb, temb))
        self.tch = ch
        self.cond = cond
        self.ada = ada
        if cond:
            self.cproj = nn.Sequential(nn.Linear(ch, temb), nn.SiLU(),
                                       nn.Linear(temb, temb))
            self.null_cond = nn.Parameter(torch.randn(temb) * 0.02)
        self.in_c = nn.Conv2d(1, ch, 3, padding=1)
        chs = [ch * m for m in mults]
        n = len(chs)

        self.dblk, self.dsamp = nn.ModuleList(), nn.ModuleList()
        c = ch
        skip_ch = []
        for i, cout in enumerate(chs):
            self.dblk.append(ResBlock(c, cout, temb, ada=ada))
            skip_ch.append(cout)                     # skip taken before downsample
            c = cout
            self.dsamp.append(nn.Conv2d(c, c, 3, stride=2, padding=1)
                              if i < n - 1 else nn.Identity())

        self.mid1 = ResBlock(c, c, temb, ada=ada)
        self.matt = Attn(c)
        self.mid2 = ResBlock(c, c, temb, ada=ada)

        self.usamp, self.ublk = nn.ModuleList(), nn.ModuleList()
        for i, cout in enumerate(reversed(chs)):
            self.usamp.append(nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
                              if i > 0 else nn.Identity())
            self.ublk.append(ResBlock(c + skip_ch.pop(), cout, temb, ada=ada))
            c = cout
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(),
                                 nn.Conv2d(c, 1, 3, padding=1))

    def _cond_emb(self, c):
        """c: (B,) float tensor; NaN entries get the learned null token."""
        null = torch.isnan(c)
        c_safe = torch.where(null, torch.zeros_like(c), c)
        emb = self.cproj(timestep_emb(c_safe, self.tch))
        return torch.where(null[:, None], self.null_cond[None], emb)

    def forward(self, x, t, c=None):
        temb = self.tproj(timestep_emb(t, self.tch))
        if self.cond and c is not None:
            temb = temb + self._cond_emb(c)
        h = self.in_c(x)
        skips = []
        for blk, ds in zip(self.dblk, self.dsamp):
            h = blk(h, temb)
            skips.append(h)                          # pre-downsample resolution
            h = ds(h)
        h = self.mid2(self.matt(self.mid1(h, temb)), temb)
        for us, blk in zip(self.usamp, self.ublk):
            h = us(h)                                # restore resolution first
            h = blk(torch.cat([h, skips.pop()], dim=1), temb)
        return self.out(h)


# --------------------------- diffusion ------------------------------------
class DDPM:
    def __init__(self, T=1000, device="cuda"):
        s = 0.008
        x = torch.linspace(0, T, T + 1)
        ac = torch.cos(((x / T + s) / (1 + s)) * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(1e-4, 0.999).to(device)
        self.T = T
        self.beta = betas
        self.ab = torch.cumprod(1 - betas, 0)
        self.device = device

    def q(self, x0, t, noise):
        ab = self.ab[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n, size, cond=None, cfg_scale=0.0):
        """Optionally condition on a scalar per sample; classifier-free
        guidance pushes samples toward the condition when cfg_scale > 0."""
        x = torch.randn(n, 1, size, size, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            if cond is not None and cfg_scale > 0:
                null = torch.full_like(cond, float("nan"))
                eps_c = model(x, t, cond)
                eps_n = model(x, t, null)
                eps = (1 + cfg_scale) * eps_c - cfg_scale * eps_n
            else:
                eps = model(x, t, cond)
            ab = self.ab[i]
            ab_p = self.ab[i - 1] if i > 0 else torch.tensor(1.0, device=self.device)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            mean = (ab_p.sqrt() * self.beta[i] / (1 - ab)) * x0 + \
                   ((1 - ab_p) * (1 - self.beta[i]).sqrt() / (1 - ab)) * x
            if i > 0:
                mean = mean + (self.beta[i] * (1 - ab_p) / (1 - ab)).sqrt() * \
                       torch.randn_like(x)
            x = mean
        return x.clamp(-1, 1)


# --------------------------- evaluation -----------------------------------
def to_physical(img, a, y_hi):
    y = (img + 1.0) / 2.0 * y_hi
    return np.sinh(np.clip(y, 0, None)) * a


def radial_profile(im):
    s = im.shape[-1]
    yy, xx = np.indices((s, s)) - s / 2
    r = np.hypot(xx, yy).astype(int)
    tbin = np.bincount(r.ravel(), im.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def power_spectrum(im):
    f = np.fft.fftshift(np.abs(np.fft.fft2(im)) ** 2)
    return radial_profile(f)


def nn_dist(A, B):
    """min L2 from each row of A to any row of B (flattened images)."""
    A = A.reshape(len(A), -1)
    B = B.reshape(len(B), -1)
    out = np.empty(len(A))
    for i in range(len(A)):
        out[i] = np.sqrt(((B - A[i]) ** 2).sum(1).min())
    return out


def evaluate(gen, train, val, attrs, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, y_hi = attrs["arcsinh_a"], attrs["y_hi"]
    gp = to_physical(gen[:, 0], a, y_hi)
    vp = to_physical(val[:, 0], a, y_hi)

    g_nn = nn_dist(gen, train)                       # generated -> train
    t_nn = nn_dist(train[:64], np.delete(train, np.arange(64), 0))  # baseline
    # diversity = nearest-neighbour distance within the generated set
    flat = gen.reshape(len(gen), -1)
    d = np.sqrt(((flat[:, None] - flat[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    g_div = d.min(1)

    # Mean-subtracted NN: removes the global-brightness coupling that breaks
    # raw L2 on heavy-zero data (where pairs of empty fields trivially have
    # small L2). Measures structural similarity rather than "how empty".
    # This is the actual memorization signal — if gen samples look like
    # specific training clusters, mean-subtracted NN drops below the
    # train-train baseline regardless of brightness.
    def _ms(x):
        f = x.reshape(len(x), -1)
        return f - f.mean(1, keepdims=True)
    g_nn_ms = nn_dist(_ms(gen), _ms(train))
    t_nn_ms = nn_dist(_ms(train[:64]), _ms(np.delete(train, np.arange(64), 0)))

    # Brightness-correlation guard: if |corr(gen mean, gen->train NN)| is
    # high, raw L2 is mostly measuring brightness and is NOT a reliable
    # memorization metric. Trust mean-subtracted NN instead. (v2 saw 0.93.)
    gen_mean = gen.reshape(len(gen), -1).mean(1)
    bright_corr = float(np.corrcoef(gen_mean, g_nn)[0, 1])
    raw_l2_reliable = abs(bright_corr) < 0.5

    rp_g = np.mean([radial_profile(x) for x in gp], 0)
    rp_v = np.mean([radial_profile(x) for x in vp], 0)
    # PSD in BOTH physical and normalized space. The a*sinh inverse is
    # violently exponential, so physical-space low-k is dominated by a
    # handful of bright pixels and exaggerates tiny errors ~10x. The
    # normalized [-1,1] space is what the model trains in and is the
    # meaningful fidelity readout; physical is kept for continuity.
    ps_g = np.mean([power_spectrum(x) for x in gp], 0)
    ps_v = np.mean([power_spectrum(x) for x in vp], 0)
    psn_g = np.mean([power_spectrum(x) for x in gen[:, 0]], 0)
    psn_v = np.mean([power_spectrum(x) for x in val[:, 0]], 0)

    # scalar trackers: low-k = mean of radial bins 1-4; DC = bin 0
    def lowk(p):
        return float(p[1:5].mean())
    lk_phys = lowk(ps_g) / lowk(ps_v)
    lk_norm = lowk(psn_g) / lowk(psn_v)
    dc_norm = float(psn_g[0] / psn_v[0])

    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    for j in range(4):
        ax[0, j].imshow(gen[j, 0], cmap="inferno", vmin=-1, vmax=1)
        ax[0, j].set_title(f"generated #{j}")
        ax[0, j].axis("off")
    ax[1, 0].plot(rp_v, label="val"); ax[1, 0].plot(rp_g, label="gen")
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title("radial profile (physical)")
    ax[1, 0].legend()
    ax[1, 1].plot(ps_v, label="val"); ax[1, 1].plot(ps_g, label="gen")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_xscale("log")
    ax[1, 1].set_title(f"power spectrum (physical)\nlow-k gen/val={lk_phys:.2f}x "
                       "(metric-inflated by sinh)")
    ax[1, 1].legend()
    ax[1, 2].plot(psn_v, label="val"); ax[1, 2].plot(psn_g, label="gen")
    ax[1, 2].set_yscale("log"); ax[1, 2].set_xscale("log")
    ax[1, 2].set_title(f"power spectrum (normalized)\nlow-k gen/val={lk_norm:.2f}x "
                       f"DC={dc_norm:.2f}x  <- meaningful")
    ax[1, 2].legend()
    ax[1, 3].hist(t_nn_ms, bins=20, alpha=0.6, density=True,
                  label="train-train (mean-sub)")
    ax[1, 3].hist(g_nn_ms, bins=20, alpha=0.6, density=True,
                  label="gen-train (mean-sub)")
    ms_ratio = float(np.median(g_nn_ms) / np.median(t_nn_ms))
    flag = "" if raw_l2_reliable else "  [raw L2 unreliable]"
    ax[1, 3].set_title(f"memorization (mean-subtracted)\n"
                       f"gen/train baseline={ms_ratio:.2f}  "
                       f"brightness-corr={bright_corr:.2f}{flag}")
    ax[1, 3].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/eval_{tag}.png", dpi=110)
    plt.close(fig)

    print(f"[eval {tag}] gen->train NN (raw L2)        median={np.median(g_nn):.3f}  "
          f"(train-train {np.median(t_nn):.3f})")
    print(f"[eval {tag}] gen->train NN (mean-subtract) median={np.median(g_nn_ms):.3f}  "
          f"(train-train {np.median(t_nn_ms):.3f})  ratio={ms_ratio:.2f}")
    print(f"[eval {tag}] gen diversity (raw L2) median={np.median(g_div):.3f}")
    print(f"[eval {tag}] low-k power gen/val: normalized={lk_norm:.2f}x "
          f"DC={dc_norm:.2f}x  | physical={lk_phys:.2f}x (sinh-inflated)")
    print(f"[eval {tag}] brightness-vs-NN corr={bright_corr:+.2f}  "
          f"-> raw L2 NN {'reliable' if raw_l2_reliable else 'UNRELIABLE (use mean-sub)'}")
    print(f"[eval {tag}]  -> MS ratio ~1.0 = good; <0.85 with bright_corr<0.5 = real memorization")


def evaluate_conditional_response(gen, cond_gen, val_imgs, val_labels,
                                  cond_scale_norm, tag):
    """Sample grid by TSC + scalar morphology trends gen vs val.

    Answers the gate question: does the model use the condition? If the
    generated mean-brightness (or radial structure) shifts with the TSC
    condition AND matches the real-data trend on the same axis, the
    conditioner takes; otherwise the generator is ignoring the condition
    and the augmentation pipeline is dead in the water.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_tsc = sorted(set(float(x) for x in cond_gen.tolist()))
    n_tsc = len(unique_tsc)
    n_per = sum(np.isclose(cond_gen, unique_tsc[0]))
    n_show = min(8, n_per)

    fig, ax = plt.subplots(n_tsc, n_show, figsize=(2 * n_show, 2 * n_tsc),
                           squeeze=False)
    for r, tsc in enumerate(unique_tsc):
        idx = np.where(np.isclose(cond_gen, tsc))[0][:n_show]
        for c, i in enumerate(idx):
            ax[r, c].imshow(gen[i, 0], cmap="inferno", vmin=-1, vmax=1)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel(f"TSC={tsc:.1f}", fontsize=10)
    fig.suptitle("Conditional samples by TSC (rows)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cond_grid_{tag}.png", dpi=110)
    plt.close(fig)

    # mean brightness vs TSC (gen) + real-data baseline if labels available
    bright_gen = gen.reshape(len(gen), -1).mean(axis=1)
    gen_means = np.array([bright_gen[np.isclose(cond_gen, t)].mean()
                          for t in unique_tsc])
    gen_stds = np.array([bright_gen[np.isclose(cond_gen, t)].std()
                          for t in unique_tsc])

    val_means = val_stds = val_centers = None
    if val_labels is not None:
        valid = ~np.isnan(val_labels)
        if valid.any():
            v_tsc_gyr = val_labels[valid] * cond_scale_norm   # back to Gyr
            v_bright = val_imgs[valid].reshape(valid.sum(), -1).mean(1)
            # bin to the same TSC grid with simple +/- half-step windows
            half = (unique_tsc[1] - unique_tsc[0]) / 2 if n_tsc > 1 else 0.5
            val_centers, val_means, val_stds = [], [], []
            for t in unique_tsc:
                m = (v_tsc_gyr >= t - half) & (v_tsc_gyr < t + half)
                if m.sum() >= 3:
                    val_centers.append(t)
                    val_means.append(float(v_bright[m].mean()))
                    val_stds.append(float(v_bright[m].std()))
            val_centers = np.asarray(val_centers)
            val_means = np.asarray(val_means)
            val_stds = np.asarray(val_stds)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    tsc_arr = np.asarray(unique_tsc)
    ax2.errorbar(tsc_arr, gen_means, yerr=gen_stds, marker="o", capsize=3,
                 label="generated", color="C1")
    if val_means is not None and len(val_means) >= 2:
        ax2.errorbar(val_centers, val_means, yerr=val_stds, marker="s",
                     capsize=3, label="real val", color="C0")
    ax2.set_xlabel("TSC (Gyr)")
    ax2.set_ylabel("mean normalized brightness")
    # correlation of gen vs condition: the gate
    corr_gen = float(np.corrcoef(cond_gen, bright_gen)[0, 1])
    ax2.set_title("Does conditioning shift sample morphology?\n"
                  f"corr(condition, gen brightness) = {corr_gen:+.3f}")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{OUT}/cond_trend_{tag}.png", dpi=110)
    plt.close(fig2)

    print(f"[cond {tag}] gen brightness by TSC bin:")
    for t, m, s in zip(unique_tsc, gen_means, gen_stds):
        print(f"  TSC={t:5.2f}  mean={m:+.3f}  std={s:.3f}")
    print(f"[cond {tag}] corr(condition, gen brightness) = {corr_gen:+.3f}")
    print(f"[cond {tag}]  -> |corr| > ~0.3 = condition is taking; "
          f"near 0 = generator ignoring it")


# ----------------------------- train --------------------------------------
def main(args):
    global OUT
    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_i, tr_l, val_i, val_l, attrs = load_split(
        args.data, seed=args.seed,
        labels_path=args.labels if args.condition else None,
        label_key=args.label_key,
        cond_scale_norm=args.cond_scale_norm,
    )
    size = tr_i.shape[-1]
    print(f"device={dev}  train={len(tr_i)} val={len(val_i)} proj-maps  size={size}")
    if args.condition:
        n_nan = int(np.isnan(tr_l).sum())
        print(f"conditional on '{args.label_key}' / {args.cond_scale_norm}  "
              f"(train: {len(tr_l) - n_nan} valid, {n_nan} NaN -> null token)")

    model = UNet(cond=args.condition, ada=args.ada).to(dev)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"UNet params: {nparam:.1f}M  conditional={args.condition}  "
          f"ada={args.ada}  sample_only={args.sample_only}")

    if args.sample_only:
        ema_path = args.load_ema or f"{OUT}/ema.pt"
        print(f"loading EMA weights from {ema_path}")
        ema = torch.load(ema_path, map_location=dev, weights_only=True)
        model.load_state_dict(ema)
        bak = ema                                    # not used in sample-only
    else:
        dl = DataLoader(RadioMaps(tr_i, tr_l, train=True, seed=args.seed),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        ddpm_train = DDPM(device=dev)
        for ep in range(args.epochs):
            model.train()
            tot = 0.0
            for batch in dl:
                if args.condition:
                    x, c = batch
                    x, c = x.to(dev), c.to(dev)
                    # CFG dropout: replace fraction of conditions with NaN
                    drop = torch.rand(c.shape[0], device=dev) < args.cond_drop_prob
                    c = torch.where(drop, torch.full_like(c, float("nan")), c)
                else:
                    x = batch.to(dev); c = None
                t = torch.randint(0, ddpm_train.T, (x.size(0),), device=dev)
                noise = torch.randn_like(x)
                pred = model(ddpm_train.q(x, t, noise), t, c)
                loss = F.mse_loss(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema[k].mul_(0.999).add_(v, alpha=0.001)
                tot += loss.item() * x.size(0)
            if ep % 20 == 0 or ep == args.epochs - 1:
                print(f"epoch {ep:4d}  loss {tot/len(tr_i):.4f}")
        bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)

    ddpm = DDPM(device=dev)
    model.eval()
    suffix = f"_{args.out_tag}" if args.out_tag else ""

    if args.condition:
        gens, conds = [], []
        for tsc_gyr in args.sample_tsc:
            cond = torch.full((args.n_per_tsc,),
                              tsc_gyr / args.cond_scale_norm, device=dev)
            g = ddpm.sample(model, args.n_per_tsc, size,
                            cond=cond, cfg_scale=args.cfg_scale).cpu().numpy()
            gens.append(g)
            conds.append(np.full(args.n_per_tsc, tsc_gyr, dtype=np.float32))
            print(f"sampled {args.n_per_tsc} at TSC={tsc_gyr:.2f}  cfg={args.cfg_scale}")
        gen = np.concatenate(gens, axis=0)
        cond_arr = np.concatenate(conds)
        np.savez(f"{OUT}/samples_cond{suffix}.npz", samples=gen, tsc=cond_arr)
        evaluate(gen, tr_i, val_i, attrs, tag=f"cond_final{suffix}")
        evaluate_conditional_response(gen, cond_arr, val_i, val_l,
                                      args.cond_scale_norm,
                                      tag=f"final{suffix}")
    else:
        gen = ddpm.sample(model, args.n_sample, size).cpu().numpy()
        np.save(f"{OUT}/samples{suffix}.npy", gen)
        evaluate(gen, tr_i, val_i, attrs, tag=f"final{suffix}")

    if not args.sample_only:
        model.load_state_dict(bak)
        torch.save(ema, f"{OUT}/ema.pt")
    print(f"done -> {OUT}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data + output
    p.add_argument("--data", type=str, default="diffusion_radio_64.h5")
    p.add_argument("--out-dir", type=str, default="diffusion_out")
    # training
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    # unconditional sampling (used when --condition is OFF)
    p.add_argument("--n-sample", type=int, default=64)
    # conditioning (TSC-scalar)
    p.add_argument("--condition", action="store_true",
                   help="train conditional on a scalar label (e.g., TSC)")
    p.add_argument("--ada", action="store_true",
                   help="use AdaGN in each ResBlock (condition modulates "
                        "per-channel scale/shift in every norm); strongly "
                        "recommended at 128px+ where the additive-time-emb "
                        "conditioning gets diluted")
    p.add_argument("--labels", type=str, default=None,
                   help="HDF5 with halo_id + label_key (e.g. TSC_eachhalo_snap99.hdf5)")
    p.add_argument("--label-key", type=str, default="tsc_gyr")
    p.add_argument("--cond-scale-norm", type=float, default=8.0,
                   help="divide labels by this so model sees ~[0,1]")
    p.add_argument("--cond-drop-prob", type=float, default=0.1,
                   help="CFG dropout: prob of replacing condition with null at train time")
    p.add_argument("--cfg-scale", type=float, default=1.5,
                   help="classifier-free guidance scale at sampling")
    p.add_argument("--sample-tsc", type=float, nargs="+",
                   default=[0.5, 1.5, 3.0, 5.0, 7.0],
                   help="TSC values (Gyr) to sample at after training")
    p.add_argument("--n-per-tsc", type=int, default=16)
    # sample-only mode: load existing EMA, skip training, re-sample at new cfg
    p.add_argument("--sample-only", action="store_true",
                   help="skip training; load EMA from --load-ema and sample only")
    p.add_argument("--load-ema", type=str, default=None,
                   help="path to ema.pt; defaults to {out-dir}/ema.pt")
    p.add_argument("--out-tag", type=str, default="",
                   help="suffix appended to all output filenames "
                        "(e.g. 'cfg3' -> samples_cond_cfg3.npz)")
    main(p.parse_args())
