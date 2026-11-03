#!/usr/bin/env python3
"""Bounded 3-D relic/event feasibility pilot; NOT a validated relic catalogue.

Select six halos by merger history alone; inspect snapshots 99 and 91.
Use the regenerated synchrotron prescription at 1.4 GHz, retaining our
n_H <= 1e-4 selection. Lee-inspired voxel/component cuts are diagnostic:
https://arxiv.org/html/2510.21632v1 (II.2, IV).
Orbit units/ordering: https://www.tng-project.org/data/cluster/#sec5g
Only two snapshots are available here; event assignments remain unverified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from scipy import ndimage
from scipy.spatial import ConvexHull, distance

import build_radio_cells as radio

BASE = Path("/oscar/data/idellant/Chuiyang")
EVENTS = "TSC_Cutimages/cluster_mergers.hdf5"


def select_sample(events, halo_ids, truth, now):
    # Verify the index->FoF mapping and cosmic clock against an independent
    # per-halo catalogue; event HaloID is NOT the actual FoF ID in this file.
    predicted = np.array([now-events["T_coll"][events["HaloID"] == i].max()
                          if np.any(events["HaloID"] == i) else np.nan
                          for i in range(len(halo_ids))])
    finite = np.isfinite(predicted) & np.isfinite(truth)
    # The legacy per-halo label uses snapshot timing for some events, whereas
    # T_coll includes interpolated pericentres. Check consistency to one cadence
    # but use continuous T_coll for this pilot, never the quantized legacy label.
    discrepancy = predicted[finite]-truth[finite]
    if not np.all((discrepancy >= -1e-5) & (discrepancy < .2)):
        raise ValueError("Event index mapping / cosmic clock failed cadence check")
    ages = now-events["T_coll"]
    records = []
    for i, halo in enumerate(halo_ids):
        ev = np.flatnonzero((events["HaloID"] == i) & (ages >= 0) & (ages <= 2)
                            & (events["Mass_ratio"] >= .1))
        if len(ev) == 1 and .15 <= ages[ev[0]] <= 1.5:
            e = int(ev[0])
            records.append(dict(halo_id=int(halo), target_index=i, event_index=e,
                                age_gyr=float(ages[e]), mass_ratio=float(events["Mass_ratio"][e])))
    frame = pd.DataFrame(records).sort_values("halo_id").reset_index(drop=True)
    rng = np.random.default_rng(42)
    chosen = []
    for lo, hi in [(.15, .5), (.5, 1.), (1., 1.50001)]:
        eligible = frame.index[(frame.age_gyr >= lo) & (frame.age_gyr < hi)].to_numpy()
        if len(eligible) < 2:
            raise ValueError(f"Fewer than two eligible halos in age bin {lo}-{hi}")
        chosen.extend(rng.choice(eligible, 2, replace=False))
    frame["selected"] = frame.index.isin(chosen)
    return frame


def largest_extent(pos):
    if len(pos) < 2:
        return 0.
    centered = pos-pos.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > max(singular[0]*1e-10, 1e-8)))
    if rank == 0:
        return 0.
    coords = centered @ vh[:rank].T
    if rank == 1:
        return float(np.ptp(coords[:, 0]))
    hull = ConvexHull(coords) if len(coords) > rank+1 else None
    vertices = coords[hull.vertices] if hull is not None else coords
    return float(distance.pdist(vertices).max())


def remove_dominant_cells(indices, weights):
    order = indices[np.argsort(weights[indices])[::-1]]
    w = weights[order]
    remaining = np.cumsum(w[::-1])[::-1]
    acceptable = np.flatnonzero(w <= .1*remaining)
    return order[acceptable[0]:] if len(acceptable) else order[:0]


def segment(pos, weights, mach, r200, logmass, voxel):
    half = np.ceil(2*r200/voxel)*voxel
    n = int(round(2*half/voxel))
    bins = np.floor((pos+half)/voxel).astype(int)
    if np.any(bins < 0) or np.any(bins >= n):
        raise ValueError("Cells outside the declared segmentation volume")
    flat = np.ravel_multi_index(bins.T, (n, n, n))
    cube = np.bincount(flat, weights=weights, minlength=n**3).reshape(n, n, n)
    labels, count = ndimage.label(cube >= 1e17, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.reshape(-1), minlength=count+1)
    ids = sorted(range(1, count+1), key=lambda i: (-int(sizes[i]), i))[:5]
    cell_labels = labels.reshape(-1)[flat]
    structures, membership = [], {}
    for lab in ids:
        original = np.flatnonzero(cell_labels == lab)
        keep = remove_dominant_cells(original, weights)
        if not len(keep):
            continue
        lum = float(weights[keep].sum())
        centroid = np.average(pos[keep], axis=0, weights=weights[keep])
        mean_mach = float(np.average(mach[keep], weights=weights[keep]))
        bright = keep[weights[keep] >= 1e19]
        lls = largest_extent(pos[bright])
        record = dict(component=int(lab), voxels=int(sizes[lab]), cells=len(keep),
                      removed_cells=len(original)-len(keep), luminosity_w_hz=lum,
                      lls_kpc=lls, weighted_mach=mean_mach,
                      x_kpc=float(centroid[0]), y_kpc=float(centroid[1]), z_kpc=float(centroid[2]),
                      radius_kpc=float(np.linalg.norm(centroid)),
                      passes_shape_cuts=bool(lls >= 300 and mean_mach <= 10))
        structures.append(record)
        membership[lab] = keep
    viable = sorted([s for s in structures if s["passes_shape_cuts"]],
                      key=lambda s: -s["luminosity_w_hz"])
    center = lambda s: np.array([s[f"{d}_kpc"] for d in "xyz"])
    pair = []
    luminosity_limit = 5e20 * (10.**logmass/1e14)**1.5
    if viable and viable[0]["luminosity_w_hz"] >= luminosity_limit:
        pair = [viable[0]]
        # Do not use collision-axis alignment or expected separation to select
        # a pair: that would build the relationship being tested into selection.
        opposite = [s for s in viable[1:]
                    if np.dot(center(s), center(pair[0])) < 0
                    and s["luminosity_w_hz"] >= luminosity_limit*.1]
        if opposite:
            pair.append(opposite[0])
    return cube, labels, half, structures, membership, pair


def metadata(snap, needed_halos):
    cat = BASE / f"targethalo_cat_TNGCluster/targethalo_cat_{snap:03d}/TargetHalo_MergerCat_{snap:03d}.hdf5"
    header = BASE / f"groupcat_classification/groupcat_{snap:03d}/fof_subhalo_tab_{snap:03d}.0.hdf5"
    with h5py.File(header) as f:
        hdr = {k: float(f["Header"].attrs[k]) for k in ["Time", "HubbleParam", "BoxSize", "Redshift"]}
    conv = hdr["Time"]/hdr["HubbleParam"]
    path = "dataset_syn_nh4_144_128.h5" if snap == 99 else "dataset_syn_nh4_144_s91_128.h5"
    with h5py.File(path) as f:
        rows = {int(h): (int(g), float(r), float(m)) for h, g, r, m in
                zip(f["meta/halo_id"][:], f["meta/halo_id_snap"][:],
                    f["meta/r500c_kpc"][:], f["meta/log_m500"][:])}
    wanted = {rows[h][0] for h in needed_halos if h in rows}
    props = {}
    if cat.exists():
        with h5py.File(cat) as f:
            ids = f["Group/FOF_Halo_IDs"][:].astype(int)
            for i in np.flatnonzero(np.isin(ids, list(wanted))):
                props[int(ids[i])] = (f["Group/GroupPos"][i]*conv,
                                      float(f["Group/Group_R_Crit200"][i])*conv,
                                      float(f["Group/Group_R_Crit500"][i])*conv)
    else:
        # Snapshot 91 has no compact target catalogue. Read only six selected
        # group rows from the chunked catalogue, never the full particle tree.
        offset = 0
        with h5py.File(header) as f:
            nfiles = int(f["Header"].attrs["NumFiles"])
        for chunk in range(nfiles):
            path = header.with_name(f"fof_subhalo_tab_{snap:03d}.{chunk}.hdf5")
            with h5py.File(path) as f:
                count = int(f["Header"].attrs["Ngroups_ThisFile"])
                for gid in sorted(wanted-props.keys()):
                    if offset <= gid < offset+count:
                        i = gid-offset
                        props[gid] = (f["Group/GroupPos"][i]*conv,
                                      float(f["Group/Group_R_Crit200"][i])*conv,
                                      float(f["Group/Group_R_Crit500"][i])*conv)
            offset += count
            if wanted <= props.keys():
                break
    if wanted-props.keys():
        raise ValueError(f"Missing selected groups at snapshot {snap}: {wanted-props.keys()}")
    return rows, props, hdr["BoxSize"]*conv, hdr["Redshift"], conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()
    out = args.run_dir / "relic_pilot"
    out.mkdir(exist_ok=False)
    with h5py.File(EVENTS) as f:
        events = {k: f[k][:] for k in ["HaloID", "T_coll", "Mass_ratio", "Collision_axis",
                                      "Mhistory_snap", "Mhistory_orbit", "Subcluster_mass"]}
    with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5") as f:
        halos, truth = f["halo_id"][:], f["tsc_gyr"][:]
    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
        if not np.array_equal(halos, f["haloID"][:]):
            raise ValueError("Merger target-index mapping differs from the TNG-Cluster catalogue")
    sample = select_sample(events, halos, truth, float(Planck15.age(0).value))
    sample.to_csv(out / "candidate_sample.csv", index=False)
    table = "Radio_Data_new/radio_spectral_ratio_table.npz"
    interp = radio.make_ratio(table)
    meta = {s: metadata(s, sample.loc[sample.selected, "halo_id"]) for s in [99, 91]}
    all_structures, all_summary, sources, event_rows = [], [], [], []
    for rec in sample[sample.selected].to_dict("records"):
        halo, event = rec["halo_id"], rec["event_index"]
        axis = events["Collision_axis"][event].astype(float)
        axis /= np.linalg.norm(axis)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for row, snap in enumerate([91, 99]):
            rows, props, box, redshift, conv = meta[snap]
            if halo not in rows:
                raise ValueError(f"Selected halo {halo} missing at snapshot {snap}")
            gid, r500, mass = rows[halo]
            center, r200, r500_cat = props[gid]
            if not np.isclose(r500, r500_cat, rtol=1e-4):
                raise ValueError("Halo centre/catalogue join failed R500 check")
            cells_dir = Path("Radio_Cells_syn_nh4_144" + ("_s91" if snap == 91 else ""))
            files = list(cells_dir.glob(f"cells_FOF{gid}_sub*.npz"))
            if len(files) != 1:
                raise ValueError(f"Expected one regenerated cell file for {gid}: {files}")
            # Use the established cutout identity, but regenerate at 1.4 GHz
            # without the 206 map-storage rescaling. Do not alter existing data.
            sub = files[0].stem.split("_sub")[-1]
            cutout = BASE / f"TNGCluster_Cutout/snap{snap}/cutout_sub{sub}_FOF{gid}.hdf5"
            before = cutout.stat()
            radio.set_redshift(redshift)
            generated = radio.generate(str(cutout), interp, max_nH=1e-4,
                                       nu_ghz=1.4, phase=True, model="synch")
            if generated is None:
                raise ValueError(f"No shock cells in {cutout}")
            pos, w, _, _, _, mach = generated
            del generated
            pos -= center
            pos -= box*np.round(pos/box)
            valid = (np.isfinite(w) & (w > 0) & np.isfinite(pos).all(axis=1)
                     & (np.linalg.norm(pos, axis=1) < 2*r200))
            source_indices = np.flatnonzero(valid)
            pos, w, mach = pos[valid], w[valid], mach[valid]
            if not len(w):
                raise ValueError(f"No valid cells in {cutout}")
            after = cutout.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Cutout changed while reading {cutout}")
            sources.append(dict(path=str(cutout), size=before.st_size, mtime_ns=before.st_mtime_ns))
            cube, labels, half, structs, members, pair = segment(pos, w, mach, r200, mass, 50.)
            voxel = 50.
            if len(pair) == 1:
                del cube, labels
                voxel = 25.
                cube, labels, half, structs, members, pair = segment(pos, w, mach, r200, mass, voxel)
            age = float(Planck15.age(redshift).value-events["T_coll"][event])
            summary = dict(halo_id=halo, snapshot=snap, event_index=event, event_age_gyr=age,
                           mass_ratio=rec["mass_ratio"], r500_kpc=r500, voxel_kpc=voxel,
                           identified_components=len(structs), pair_members=len(pair),
                           association_status="unverified" if age >= 0 else "selected_event_not_yet_occurred")
            cent = lambda s: np.array([s[f"{d}_kpc"] for d in "xyz"])
            if len(pair) == 2:
                vector = cent(pair[0])-cent(pair[1])
                separation = float(np.linalg.norm(vector))
                summary.update(separation_kpc=separation, separation_r500=separation/r500,
                               collision_axis_angle_deg=float(np.degrees(np.arccos(
                                   np.clip(abs(np.dot(vector/separation, axis)), 0, 1)))))
            all_summary.append(summary)
            for st in structs:
                all_structures.append(dict(halo_id=halo, snapshot=snap, **st))
            packed = {}
            for lab, idx in members.items():
                packed.update({f"component_{lab}_generated_cell_index": source_indices[idx],
                               f"component_{lab}_position_kpc": pos[idx],
                               f"component_{lab}_luminosity_w_hz": w[idx]})
            np.savez_compressed(out / f"members_{halo}_snap{snap}.npz", **packed)
            for ax, (i, j, los) in zip(axes[row], [(0, 1, 2), (1, 2, 0), (0, 2, 1)]):
                projection = cube.sum(axis=los)
                im = ax.imshow(np.log10(np.maximum(projection.T, 1e15)), origin="lower",
                               extent=np.array([-half, half, -half, half])/r500, cmap="inferno")
                ax.plot([-2*axis[i], 2*axis[i]], [-2*axis[j], 2*axis[j]], "w--", lw=.8)
                for st, color in zip(pair, ["cyan", "lime"]):
                    mask = (labels == st["component"]).any(axis=los).T
                    grid = (np.arange(mask.shape[0])+.5)*voxel-half
                    ax.contour(grid/r500, grid/r500, mask, levels=[.5], colors=[color], linewidths=.7)
                    p = cent(st)/r500
                    ax.scatter(p[i], p[j], marker="x", color=color)
                # Recorded main/subcluster positions at this snapshot, if valid.
                times = np.flatnonzero(events["Mhistory_snap"][event] == snap)
                if len(times):
                    orbit = events["Mhistory_orbit"][event, times[0]].reshape(2, 3)*conv-center
                    orbit -= box*np.round(orbit/box)
                    ax.scatter(orbit[:, i]/r500, orbit[:, j]/r500, marker="+", color="white")
                ax.set(xlabel=f"{'xyz'[i]} / R500", ylabel=f"{'xyz'[j]} / R500",
                       title=f"snap {snap}, event age {age:+.2f} Gyr, {voxel:g} kpc bins")
            fig.colorbar(im, ax=list(axes[row]), label="log10 projected L1.4 [W/Hz per bin]", shrink=.8)
            del cube, labels, pos, w, mach
            print(json.dumps(summary), flush=True)
        fig.suptitle(f"Halo {halo}: candidate event {event}; association NOT validated")
        fig.savefig(out / f"halo_{halo}.pdf")
        plt.close(fig)
        for e in np.flatnonzero(events["HaloID"] == rec["target_index"]):
            event_rows.append(dict(halo_id=halo, event_index=int(e),
                                   selected_event=bool(e == event),
                                   age_at_z0_gyr=float(Planck15.age(0).value-events["T_coll"][e]),
                                   mass_ratio=float(events["Mass_ratio"][e]),
                                   subcluster_mass_1e14_msun=float(events["Subcluster_mass"][e])))
    pd.DataFrame(all_summary).to_csv(out / "structure_event_candidates.csv", index=False)
    pd.DataFrame(all_structures).to_csv(out / "structures.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out / "competing_events.csv", index=False)
    manifest = dict(selected_halos=sample.loc[sample.selected, "halo_id"].tolist(),
                    sample_rule="one mass-ratio>=0.1 event in last 2 Gyr; age 0.15-1.5 Gyr; two per age bin; seed 42",
                    snapshots=[91, 99], radio_model="synch", frequency_ghz=1.4, max_nh=1e-4,
                    spectral_table_sha256=hashlib.sha256(Path(table).read_bytes()).hexdigest(),
                    inputs=sources,
                    limitations=["Two snapshots do not establish event association; manual inspection required.",
                                 "Current synchrotron prescription and density cut differ from a full Lee reproduction.",
                                 "No population slope or timing accuracy is inferred from this selected six-halo sample.",
                                 "Existing cutout spatial/zoom coverage is assumed; verify before a full catalogue."],
                    sources=["https://arxiv.org/html/2510.21632v1", "https://www.tng-project.org/data/cluster/"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Pilot artifacts ready for visual inspection", flush=True)


if __name__ == "__main__":
    main()
