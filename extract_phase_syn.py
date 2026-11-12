#!/usr/bin/env python3
"""Stale-regeneration W3 (review rounds 48-49): gas state and frequency
behaviour of the CURRENT (synch, 144 MHz) radio weights, per shock cell, from
the raw snapshot-99 cutouts. extract_phase.py is the historical (HB07) recipe
and is left untouched.

Cell population: every shock cell selected by build_radio_cells.generate
(Machnumber > 1.3 and EnergyDissipation > 0), no density cut, z = 0, synch
model, ratio table as in submit_syn_cells.sh. A cell is VALID if its weight is
finite and >= 0 and n_H, kT are finite and > 0; invalid cells are counted, not
binned. Zero-weight valid cells count in the number histogram.

Per parent (normalised by the parent's whole valid population, NOT the
in-range one; out-of-range number and weight are saved):
  H_n, H_w       (log n_H, log kT) histograms, number and 144 MHz weight
  frac_above     weight fraction with n_H > 1e-4 (the cut keeps n_H <= 1e-4)
  frac_cold      weight fraction with kT < 0.1 keV (declared cold threshold)
  top_all/_cut   brightest-CELL weight share, all valid cells / kept cells
  mach_all/_cut  weight-averaged Mach number, all / kept
Frequency (kept cells; unscaled physical weights): w1400 = w144 *
(1.4/0.144)^(-s/2) with s = s_of_M(M), the implemented synch law at fixed
cell properties. ratio_cut = sum w144 / sum w1400 per parent (also for all
valid cells). Checks against the stored cell files: kept-cell count and
sum(w144)/206 must match Radio_Cells_syn_nh4_144; all-cell count must match
Radio_Cells_syn_144.

  extract_phase_syn.py --out <file.npz>
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import time

import numpy as np

import build_radio_cells as brc

C = "/oscar/data/idellant/Chuiyang"
NH_EDGES = np.linspace(-7.0, 0.0, 71)        # log10 n_H [cm^-3]
T_EDGES = np.linspace(-3.5, 1.5, 51)         # log10 kT [keV]
NH_CUT, T_COLD_KEV, W_SCALE = 1e-4, 0.1, 206.0
NU_LO, NU_HI = 0.144, 1.4


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parent_stats(w, T, nH, M):
    """All per-parent quantities from one parent's cells (pure function)."""
    w, T, nH, M = (np.asarray(v, np.float64) for v in (w, T, nH, M))
    valid = np.isfinite(w) & (w >= 0) & np.isfinite(nH) & (nH > 0) & np.isfinite(T) & (T > 0) & np.isfinite(M)
    out = dict(n_cells=len(w), n_invalid=int((~valid).sum()))
    w, T, nH, M = w[valid], T[valid], nH[valid], M[valid]
    x, y = np.log10(nH), np.log10(T)
    inr = (x >= NH_EDGES[0]) & (x <= NH_EDGES[-1]) & (y >= T_EDGES[0]) & (y <= T_EDGES[-1])
    W = w.sum()
    hn, _, _ = np.histogram2d(x, y, [NH_EDGES, T_EDGES])
    hw, _, _ = np.histogram2d(x, y, [NH_EDGES, T_EDGES], weights=w)
    keep = nH <= NH_CUT
    out.update(n_valid=len(w), n_zero_w=int((w == 0).sum()), w_total=float(W),
               n_out_of_range=int((~inr).sum()), w_out_of_range=float(w[~inr].sum()),
               H_n=hn / len(w) if len(w) else hn * np.nan,
               H_w=hw / W if W > 0 else hw * np.nan,
               n_kept=int(keep.sum()), w_kept=float(w[keep].sum()))
    if W > 0:
        out.update(frac_above=float(w[~keep].sum() / W), frac_cold=float(w[T < T_COLD_KEV].sum() / W),
                   top_all=float(w.max() / W), mach_all=float((w * M).sum() / W))
    else:
        out.update(frac_above=np.nan, frac_cold=np.nan, top_all=np.nan, mach_all=np.nan)
    wk = w[keep]
    if wk.sum() > 0:
        out.update(top_cut=float(wk.max() / wk.sum()), mach_cut=float((wk * M[keep]).sum() / wk.sum()),
                   cut_status="ok")
    else:
        out.update(top_cut=np.nan, mach_cut=np.nan,
                   cut_status="empty cut" if not keep.any() else "zero weight after cut")
    fac = (NU_HI / NU_LO) ** (-0.5 * brc.s_of_M(M))
    w14 = w * fac
    out["ratio_all"] = float(W / w14.sum()) if w14.sum() > 0 else np.nan
    out["ratio_cut"] = float(wk.sum() / w14[keep].sum()) if w14[keep].sum() > 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--radio-dir", default=f"{C}/Radio_Snap99_new")
    ap.add_argument("--cutout-dir", default=f"{C}/TNGCluster_Cutout/snap99")
    ap.add_argument("--ratio-table", default="Radio_Data_new/radio_spectral_ratio_table.npz")
    ap.add_argument("--cells-cut", default="Radio_Cells_syn_nh4_144")
    ap.add_argument("--cells-all", default="Radio_Cells_syn_144")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not a.out.endswith(".npz"):
        raise SystemExit("--out must end in .npz")
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "extract_phase_syn.py", "build_radio_cells.py"],
                           cwd=here, capture_output=True, text=True).stdout.strip()
    brc.set_redshift(0.0)
    interp = brc.make_ratio(a.ratio_table)
    srcs = sorted(glob.glob(os.path.join(a.radio_dir, "radio_FOF*.npz")))
    if a.limit:
        srcs = srcs[:a.limit]
    keys = ["halo_id", "n_cells", "n_invalid", "n_valid", "n_zero_w", "w_total", "n_out_of_range",
            "w_out_of_range", "n_kept", "w_kept", "frac_above", "frac_cold", "top_all", "top_cut",
            "mach_all", "mach_cut", "ratio_all", "ratio_cut", "check_cut_n", "check_cut_wrel", "check_all_n"]
    rec = {k: [] for k in keys}
    status, Hn, Hw = [], [], []
    t0 = time.time()
    for i, src in enumerate(srcs):
        fof, sub = re.search(r"radio_FOF(\d+)_sub(\d+)\.npz", os.path.basename(src)).groups()
        cut = os.path.join(a.cutout_dir, f"cutout_sub{sub}_FOF{fof}.hdf5")
        if not os.path.exists(cut):
            raise SystemExit(f"missing cutout for FOF{fof}")
        out = brc.generate(cut, interp, nu_ghz=NU_LO, phase=True, model="synch")
        if out is None:
            raise SystemExit(f"FOF{fof}: no shock cells (the cell build had none missing)")
        _, w, _, T, nH, M = out
        st = parent_stats(w, T, nH, M)
        cc = np.load(os.path.join(a.cells_cut, f"cells_FOF{fof}_sub{sub}.npz"))["w"]
        ca = np.load(os.path.join(a.cells_all, f"cells_FOF{fof}_sub{sub}.npz"))["w"]
        # checks on the UNFILTERED arrays: the cell build keeps n_H <= 1e-4
        # (and T >= 0) without any validity filter
        raw_keep = np.asarray(nH, np.float64) <= NH_CUT
        st["check_cut_n"] = int(raw_keep.sum() - len(cc))
        st["check_all_n"] = int(len(w) - len(ca))
        st["check_cut_wrel"] = float(abs(np.asarray(w, np.float64)[raw_keep].sum() / W_SCALE - cc.sum())
                                     / cc.sum())
        st["halo_id"] = int(fof)
        Hn.append(st.pop("H_n")); Hw.append(st.pop("H_w"))
        status.append(st.pop("cut_status"))
        for k in keys:
            rec[k].append(st[k])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(srcs)} ({time.time() - t0:.0f}s)", flush=True)
    arr = {k: np.array(v) for k, v in rec.items()}
    bad = (arr["check_cut_n"] != 0) | (arr["check_all_n"] != 0) | ~(arr["check_cut_wrel"] <= 1e-9)
    prov = dict(head=head, uncommitted=dirty, args=vars(a), w_scale_cells=W_SCALE, nh_cut=NH_CUT,
                t_cold_kev=T_COLD_KEV, nu=[NU_LO, NU_HI], z=0.0, shock="Machnumber > 1.3 & EnergyDissipation > 0",
                normalisation="per parent by its whole valid population",
                sources={f: sha(os.path.join(here, f)) for f in ("extract_phase_syn.py", "build_radio_cells.py")},
                ratio_table=sha(a.ratio_table), n_parents=len(srcs), n_check_failures=int(bad.sum()),
                seconds=round(time.time() - t0, 1))
    np.savez(a.out, H_n=np.array(Hn), H_w=np.array(Hw), nh_edges=NH_EDGES, t_edges=T_EDGES,
             cut_status=np.array(status), provenance=np.array(json.dumps(prov)), **arr)
    print(json.dumps({k: v for k, v in prov.items() if k != "args"}, indent=1))
    if bad.any():
        raise SystemExit(f"cell-file consistency check FAILED for {int(bad.sum())} parents "
                         f"(results saved for diagnosis in {a.out})")
    print(f"saved {a.out}: {len(srcs)} parents; all cell-file checks passed")


if __name__ == "__main__":
    main()
