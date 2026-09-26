"""How big would an earlier-snapshot download be if we fetched only the gas
fields the radio and X-ray mocks need?  Measured on existing snap-99 cutouts
(read-only, Chuiyang's directory)."""
import glob, os
import h5py
import numpy as np

RADIO = ["Coordinates", "Masses", "Density", "Machnumber", "EnergyDissipation",
         "MagneticField", "InternalEnergy", "ElectronAbundance",
         "StarFormationRate"]
XRAY = ["Coordinates", "Masses", "Density", "InternalEnergy",
        "ElectronAbundance", "GFM_Metallicity", "StarFormationRate",
        "GFM_CoolingRate", "Velocities"]
files = sorted(glob.glob("/oscar/data/idellant/Chuiyang/TNGCluster_Cutout/snap99/cutout_*.hdf5"))
rng = np.random.default_rng(0)
pick = [files[i] for i in rng.choice(len(files), 8, replace=False)]
tot_file, tot_need, tot_gas = 0, 0, 0
for p in pick:
    size = os.path.getsize(p)
    need = gas = 0
    with h5py.File(p, "r") as f:
        g = f["PartType0"]
        for k in g:
            nb = g[k].size * g[k].dtype.itemsize
            gas += nb
            if k in set(RADIO) | set(XRAY):
                need += nb
        types = {t: f[t]["Coordinates"].shape[0] for t in f if t.startswith("PartType")}
    tot_file += size; tot_need += need; tot_gas += gas
    print(f"{os.path.basename(p):<40} file {size/1e9:6.2f} GB  gas {gas/1e9:6.2f}  "
          f"needed fields {need/1e9:6.2f}  particles {types}")
n = len(files)
print(f"\n{n} cutouts on disk, total {sum(os.path.getsize(p) for p in files)/1e9:.0f} GB")
print(f"needed/total over sample: {tot_need/tot_file:.2%}; gas/total {tot_gas/tot_file:.2%}")
print(f"estimated per snapshot, needed gas fields only: "
      f"{tot_need/tot_file*sum(os.path.getsize(p) for p in files)/1e9:.0f} GB")
