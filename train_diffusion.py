#!/usr/bin/env python3
"""
Unconditional DDPM on radio cluster maps — a decisive memorization test.

With ~350 independent clusters the question is not "can it make plausible
maps" but "plausible AND novel". This script trains a small UNet DDPM and
reports the three readouts that actually decide feasibility:

  1. held-out fidelity   — physical radial profile + 1D power spectrum of
                            generated vs. validation maps (stretch inverted)
  2. memorization        — NN L2 distance generated->train, compared to the
                            train->train NN baseline (collapse => failure)
  3. diversity           — pairwise distance spread among generated samples

Cluster-level split (3 projections per halo kept in the same fold) reuses the
project's no-leakage rule. Outputs go to diffusion_out/.

Usage (via submit_diffusion.sh on the gpu partition):
  python train_diffusion.py --data diffusion_radio_64.h5 --epochs 400
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
    """Projection-level grayscale maps; 8x rot/flip aug (physically valid)."""

    def __init__(self, imgs, train=True, seed=0):
        self.imgs = imgs  # (M, 1, S, S) float32 in [-1, 1]
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
        return torch.from_numpy(x)


def load_split(path, val_frac=0.15, seed=0):
    with h5py.File(path, "r") as f:
        imgs = f["images"][:]                       # (N, 3, S, S)
        halo = f["meta/halo_id"][:]
        attrs = dict(f.attrs)
    rng = np.random.default_rng(seed)
    uniq = np.unique(halo)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac)
    val_h = set(uniq[:n_val].tolist())
    tr_m = np.array([h not in val_h for h in halo])
    # expand (N,3,S,S) -> (N*3,1,S,S) at projection level
    def expand(mask):
        sel = imgs[mask]                            # (n, 3, S, S)
        return sel.reshape(-1, 1, *sel.shape[2:]).astype(np.float32)
    return expand(tr_m), expand(~tr_m), attrs


# --------------------------- model ----------------------------------------
def timestep_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
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
    then strided downsample; mirror on the way up with transpose-conv."""

    def __init__(self, ch=64, mults=(1, 2, 4)):
        super().__init__()
        temb = ch * 4
        self.tproj = nn.Sequential(nn.Linear(ch, temb), nn.SiLU(),
                                   nn.Linear(temb, temb))
        self.tch = ch
        self.in_c = nn.Conv2d(1, ch, 3, padding=1)
        chs = [ch * m for m in mults]
        n = len(chs)

        self.dblk, self.dsamp = nn.ModuleList(), nn.ModuleList()
        c = ch
        skip_ch = []
        for i, cout in enumerate(chs):
            self.dblk.append(ResBlock(c, cout, temb))
            skip_ch.append(cout)                     # skip taken before downsample
            c = cout
            self.dsamp.append(nn.Conv2d(c, c, 3, stride=2, padding=1)
                              if i < n - 1 else nn.Identity())

        self.mid1 = ResBlock(c, c, temb)
        self.matt = Attn(c)
        self.mid2 = ResBlock(c, c, temb)

        self.usamp, self.ublk = nn.ModuleList(), nn.ModuleList()
        for i, cout in enumerate(reversed(chs)):
            self.usamp.append(nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
                              if i > 0 else nn.Identity())
            self.ublk.append(ResBlock(c + skip_ch.pop(), cout, temb))
            c = cout
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(),
                                 nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.tproj(timestep_emb(t, self.tch))
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
    def sample(self, model, n, size):
        x = torch.randn(n, 1, size, size, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = model(x, t)
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
    ax[1, 3].hist(t_nn, bins=20, alpha=0.6, density=True, label="train-train NN")
    ax[1, 3].hist(g_nn, bins=20, alpha=0.6, density=True, label="gen-train NN")
    ax[1, 3].set_title("memorization check"); ax[1, 3].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/eval_{tag}.png", dpi=110)
    plt.close(fig)

    print(f"[eval {tag}] gen->train NN  median={np.median(g_nn):.3f}  "
          f"(train-train baseline median={np.median(t_nn):.3f})")
    print(f"[eval {tag}] gen diversity (min pairwise) median={np.median(g_div):.3f}")
    print(f"[eval {tag}] low-k power gen/val: normalized={lk_norm:.2f}x "
          f"DC={dc_norm:.2f}x  | physical={lk_phys:.2f}x (sinh-inflated)")
    print(f"[eval {tag}]  -> track NORMALIZED low-k (~1.0 = good); "
          f"memorization risk if gen->train << train-train baseline")


# ----------------------------- train --------------------------------------
def main(args):
    global OUT
    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, val, attrs = load_split(args.data, seed=args.seed)
    size = tr.shape[-1]
    print(f"device={dev}  train={len(tr)} val={len(val)} proj-maps  size={size}")

    dl = DataLoader(RadioMaps(tr, train=True, seed=args.seed),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=4, drop_last=True)
    model = UNet().to(dev)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ddpm = DDPM(device=dev)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"UNet params: {nparam:.1f}M")

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x in dl:
            x = x.to(dev)
            t = torch.randint(0, ddpm.T, (x.size(0),), device=dev)
            noise = torch.randn_like(x)
            pred = model(ddpm.q(x, t, noise), t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(0.999).add_(v, alpha=0.001)
            tot += loss.item() * x.size(0)
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:4d}  loss {tot/len(tr):.4f}")

    bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema)
    model.eval()
    gen = ddpm.sample(model, args.n_sample, size).cpu().numpy()
    np.save(f"{OUT}/samples.npy", gen)
    evaluate(gen, tr, val, attrs, tag="final")
    model.load_state_dict(bak)
    torch.save(ema, f"{OUT}/ema.pt")
    print(f"done -> {OUT}/ (samples.npy, ema.pt, eval_final.png)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="diffusion_radio_64.h5")
    p.add_argument("--out-dir", type=str, default="diffusion_out")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-sample", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
