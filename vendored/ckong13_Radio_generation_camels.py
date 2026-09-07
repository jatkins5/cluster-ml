#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import re
import h5py
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.constants import k_B, m_e
from scipy.interpolate import RegularGridInterpolator
#import matplotlib.pyplot as plt


# In[ ]:


cutout_dir = Path(
    "/users/ckong13/data/Chuiyang/Camels/Cutout_Snap90"
)

radio_dir = Path(
    "/users/ckong13/data/Chuiyang/Camels/Radio"
)
radio_dir.mkdir(parents=True, exist_ok=True)

psi_table_file = Path("/users/ckong13/data/Chuiyang/Camels/Radio/normalized_psi_table.npz")

snap_num = 90
radius_factor = 2.0
mach_min = 1.3
overwrite = False


# In[ ]:


# ============================================================
# Load the normalized psi table
# ============================================================

if not psi_table_file.exists():
    raise FileNotFoundError(
        f"Psi table not found: {psi_table_file.resolve()}"
    )

data = np.load(psi_table_file)

s_grid = data["s_grid"]
loge_grid = data["loge_grid"]     # log10(emin)
normalized_psi_grid = data["normalized_psi_grid"]

interp_psi = RegularGridInterpolator(
    (s_grid, loge_grid),
    normalized_psi_grid,
    bounds_error=False,
    fill_value=None,
)


# In[ ]:


# ============================================================
# Radio-model helper functions
# ============================================================

GAMMA = 5/3
XH = 0.76
KPC_IN_CM = u.kpc.to(u.cm)

def r_of_M(M, gamma=GAMMA):
    M2 = M*M
    return (gamma + 1.0) * M2 / ((gamma - 1.0) * M2 + 2.0)

def s_of_M(M, gamma=GAMMA):
    r = r_of_M(M, gamma)
    s = (r+2)/(r-1)
    return s

def emin_of_TkeV(T_keV):
    e_min_vals = 10*T_keV/511
    return  e_min_vals
    
def vecnorm(B):
    return np.sqrt((B**2).sum(axis=1))

def Bcmb_uG(z):
    # B_CMB ≈ 3.24 (1+z)^2 microGauss
    return 3.24 * (1.0 + z)**2

def phi_M(M, T_keV, gamma=GAMMA):
    s = s_of_M(M, gamma)
    e_min_vals = emin_of_TkeV(T_keV)
    log_e_min_vals = np.log10(e_min_vals)

    pts = np.column_stack([s, log_e_min_vals])
    phi = interp_psi(pts)
    return phi

def zoom_number_from_path(path):
    match = re.search(r"GZ28_(\d+)_snap", path.name)

    if match is None:
        raise ValueError(
            f"Cannot extract zoom number from {path.name}"
        )

    return int(match.group(1))


# In[ ]:


# ============================================================
# Generate radio-emitting shock cells for one CAMELS cutout
# ============================================================

def generate_radio_camels(
    cutout_name,
    mach_min=1.3,
    verbose=True,
):
    cutout_name = Path(cutout_name)

    with h5py.File(cutout_name, "r") as f:
        header = f["Header"].attrs

        # Cosmology and code units for this individual CAMELS zoom
        a = float(header["Time"])
        z = float(header["Redshift"])
        h = float(header["HubbleParam"])

        unit_length_cm = float(header["UnitLength_in_cm"])
        unit_mass_g = float(header["UnitMass_in_g"])
        unit_velocity_cms = float(header["UnitVelocity_in_cm_per_s"])

        if "CenteredCoordinates" not in f["PartType0"]:
            raise KeyError(
                "PartType0/CenteredCoordinates is missing. "
                "Use the CAMELS cutouts made by make_cutout.py."
            )

        centered_coordinates = (f["PartType0/CenteredCoordinates"][:])
        electron_abundance = (f["PartType0/ElectronAbundance"][:])
        internal_energy = (f["PartType0/InternalEnergy"][:])
        mach_number = (f["PartType0/Machnumber"][:])
        magnetic_field = (f["PartType0/MagneticField"][:])
        energy_dissipation = (f["PartType0/EnergyDissipation"][:])

        if "GroupInfo/Group_R_Crit200" in f:
            r200_code = float(
                f["GroupInfo/Group_R_Crit200"][()]
            )
        else:
            r200_code = np.nan

        zoom_number = int(header["ZoomNumber"])
        snap_number = int(header["SnapNum"])

    # --------------------------------------------------------
    # Select shock cells
    # --------------------------------------------------------

    shock = (
        np.isfinite(mach_number)
        & np.isfinite(energy_dissipation)
        & np.isfinite(internal_energy)
        & np.isfinite(electron_abundance)
        & (mach_number > mach_min)
        & (energy_dissipation > 0.0)
    )

    if not np.any(shock):
        metadata = {
            "zoom_number": zoom_number,
            "snap_num": snap_number,
            "a": a,
            "z": z,
            "h": h,
            "r200_kpc": r200_code
            * (unit_length_cm / KPC_IN_CM)
            * a / h,
            "number_of_shock_cells": 0,
        }

        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            metadata,
        )

    # --------------------------------------------------------
    # Temperature
    # InternalEnergy is in code velocity^2 units.
    # --------------------------------------------------------
    
    # the mean molecular weight
    mu_g = (4.0/(1.0+ 3.0 * XH+ 4.0 * XH * electron_abundance[shock])* const.m_p.cgs.value)
    
    internal_energy_cgs = (internal_energy[shock].astype(np.float64)* unit_velocity_cms**2)

    temperature_K = ((GAMMA - 1.0)* internal_energy_cgs* mu_g/ const.k_B.cgs.value)

    T_keV = (k_B * (temperature_K * u.K)).to(u.keV).value

    # --------------------------------------------------------
    # CAMELS code units -> physical quantities
    # --------------------------------------------------------

    unit_length_kpc = unit_length_cm / KPC_IN_CM

    pos_kpc = (
        centered_coordinates[shock].astype(np.float64)
        * unit_length_kpc
        * a / h
    )

    M = mach_number[shock].astype(np.float64)

    # Same EnergyDissipation conversion used in the original
    # TNG-Cluster notebook, but derived from this file's units.
    # convert to E/(1e44 erg/s)
    
    unit_edot_erg_s = (
        unit_mass_g / unit_length_cm
    ) * unit_velocity_cms**3

    E_1e44 = (
        energy_dissipation[shock].astype(np.float64)
        / a
        * unit_edot_erg_s
        * 1.0e-44
    )

    # Magnetic-field code unit:
    # sqrt(M/L) * V/L, followed by the original h/a^2 scaling.
    unit_B_gauss = (
        np.sqrt(unit_mass_g / unit_length_cm)
        * unit_velocity_cms
        / unit_length_cm
    )
    
    #convert unit of B to \muG
    B_uG = (
        vecnorm(magnetic_field[shock].astype(np.float64))
        * h / a**2
        * unit_B_gauss
        * 1.0e6
    )

    s = s_of_M(M)
    Bcmb = Bcmb_uG(z)
    phi = phi_M(M, T_keV)

    w = (
        5.2e23
        * E_1e44
        * B_uG**(1.0 + 0.5 * s)
        / (B_uG**2 + Bcmb**2)
        * phi
    )

    valid = (
        np.all(np.isfinite(pos_kpc), axis=1)
        & np.isfinite(w)
        & (w >= 0.0)
    )

    pos_kpc = pos_kpc[valid]
    w = w[valid]

    r200_kpc = (
        r200_code
        * unit_length_kpc
        * a / h
    )

    metadata = {
        "zoom_number": zoom_number,
        "snap_num": snap_number,
        "a": a,
        "z": z,
        "h": h,
        "r200_kpc": r200_kpc,
        "number_of_shock_cells": len(w),
    }

    if verbose:
        print(
            f"GZ28_{zoom_number}: "
            f"z={z:.6g}, h={h:.6g}, "
            f"shock cells={len(w)}"
        )

    return pos_kpc, w, metadata


# In[ ]:


# ============================================================
# Find completed CAMELS cutouts
# ============================================================

cutout_files = sorted(
    cutout_dir.glob(
        f"GZ28_*_snap{snap_num:03d}_{radius_factor:g}R200.hdf5"
    ),
    key=zoom_number_from_path,
)

print("Completed cutouts found:", len(cutout_files))
print("First five:", [p.name for p in cutout_files[:5]])
print("Last five:", [p.name for p in cutout_files[-5:]])



# ============================================================
# Generate radio data for every completed CAMELS cutout
# ============================================================

successful = []
failed = {}

for i, cutout_file in enumerate(cutout_files):
    zoom_number = zoom_number_from_path(cutout_file)

    output_file = (
        radio_dir
        / f"radio_GZ28_{zoom_number}_snap{snap_num:03d}.hdf5"
    )

    temporary_file = output_file.with_name(
        output_file.stem + ".tmp.hdf5"
    )

    print(
        f"\n[{i + 1:3d}/{len(cutout_files):3d}] "
        f"GZ28_{zoom_number}"
    )

    if output_file.exists() and not overwrite:
        print(f"[SKIP] {output_file.name} already exists")
        successful.append(zoom_number)
        continue

    if temporary_file.exists():
        temporary_file.unlink()

    try:
        pos, w, metadata = generate_radio_camels(
            cutout_file,
            mach_min=mach_min,
            verbose=True,
        )

        with h5py.File(temporary_file, "w") as f_out:

            # Simulation and halo metadata
            header = f_out.create_group("Header")

            header.attrs["ZoomNumber"] = int(
                metadata["zoom_number"]
            )
            header.attrs["SnapNum"] = int(
                metadata["snap_num"]
            )
            header.attrs["Time"] = float(
                metadata["a"]
            )
            header.attrs["Redshift"] = float(
                metadata["z"]
            )
            header.attrs["HubbleParam"] = float(
                metadata["h"]
            )
            header.attrs["R200_kpc"] = float(
                metadata["r200_kpc"]
            )
            header.attrs["NumberOfShockCells"] = int(
                metadata["number_of_shock_cells"]
            )
            header.attrs["MachMin"] = float(mach_min)
            header.attrs["SourceCutout"] = cutout_file.name

            # Radio-emitting shock-cell data
            radio = f_out.create_group("Radio")

            pos_array = np.asarray(pos, dtype=np.float64)
            w_array = np.asarray(w, dtype=np.float64)

            pos_dataset = radio.create_dataset(
                "Coordinates",
                data=pos_array,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            w_dataset = radio.create_dataset(
                "Power",
                data=w_array,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            pos_dataset.attrs["Unit"] = "physical kpc"
            pos_dataset.attrs[
                "Description"
            ] = "Shock-cell coordinates relative to the halo center"

            w_dataset.attrs[
                "Description"
            ] = "Radio weight returned by generate_radio_camels"

            temporary_file.replace(output_file)
            successful.append(zoom_number)

    except BaseException as error:
        if temporary_file.exists():
            temporary_file.unlink()

        failed[zoom_number] = str(error)
        print(f"[FAILED] GZ28_{zoom_number}: {error}")


print("\n" + "=" * 70)
print("Radio generation finished")
print("=" * 70)
print("Successful:", len(successful))
print("Failed:", len(failed))

if failed:
    print("\nFailed zooms:")
    for zoom_number, error in failed.items():
        print(f"GZ28_{zoom_number}: {error}")

