"""What optical inputs exist on disk for review experiments E1/E2 (read-only)."""
import glob, os, h5py
C = "/oscar/data/idellant/Chuiyang"
f0 = sorted(glob.glob(f"{C}/groupcat_classification/groupcat_099/fof_subhalo_tab_099.*.hdf5"))
print(f"groupcat_099 chunks: {len(f0)}")
with h5py.File(f0[0]) as f:
    sub = [k for k in f["Subhalo"]] if "Subhalo" in f else []
    print("  Subhalo fields:", sorted(sub))
    if "SubhaloStellarPhotometrics" in f.get("Subhalo", {}):
        print("  SubhaloStellarPhotometrics shape", f["Subhalo/SubhaloStellarPhotometrics"].shape)
print("BCG_offset_Cal:")
for p in sorted(glob.glob(f"{C}/BCG_offset_Cal/*"))[:20]:
    print("  ", os.path.basename(p), os.path.getsize(p) if os.path.isfile(p) else "dir")
cut = sorted(glob.glob(f"{C}/TNGCluster_Cutout/snap99/cutout_*.hdf5"))[0]
with h5py.File(cut) as f:
    print("cutout particle types:", list(f.keys()))
    if "PartType4" in f:
        print("  PartType4 fields:", sorted(f["PartType4"].keys()))
