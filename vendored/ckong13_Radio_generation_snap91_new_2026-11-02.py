#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.constants import k_B, m_e
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# In[2]:


# Cosmic parameters
Snap_Index = 91

from astropy.cosmology import Planck15
h0 = Planck15.h


# In[ ]:


# unit conversion

# compute the numerical factor in cgs from the units of B in TNG-Cluster
expr = np.sqrt(1e10 * u.Msun / u.kpc) * (u.km/u.s) / u.kpc

# convert the expression to sqrt(dyne/cm^2) (i.e. sqrt(pressure) in cgs)
# since 1 dyne/cm^2 = 1 g/(cm s^2)
expr_cgs = expr.to(u.g**0.5 / (u.cm**0.5 * u.s))   # same as sqrt(dyne/cm^2)

UnitB_G = expr_cgs.value

Econv = ((1e10 * u.Msun / u.kpc) * (u.km/u.s)**3).to(u.erg/u.s).value

# Model parameters
XI_E = 0.05
NU_MHZ = 144.0
NU_REF_MHZ = 144.0

# Reference coefficient: numerical value in W/Hz
POWER_PREF_W_HZ = 8.701249739473983e+34

# Dimensionless reference-frequency factor
SPECTRAL_BASE = 2.290130219242185e-08


# In[ ]:


# define the psi function using interpolation

table_path = "./radio_spectral_ratio_table.npz"
with np.load(table_path) as data:
    s_grid = data["s_grid"]
    loge_grid = data["loge_grid"]
    log10_spectral_ratio = data["log10_spectral_ratio"]
    E_MAX = float(data["e_max"])
    TABLE_GAMMA = float(data["gamma"])

# Interpolate log10(I_Psi / I_spec)
interp_log_ratio = RegularGridInterpolator(
    (s_grid, loge_grid),
    log10_spectral_ratio,
    bounds_error=True,
)


# In[ ]:


# ---------- helpers ----------
GAMMA = 5/3
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

def spectral_ratio_MT(M, T_keV, gamma=GAMMA):
    # Convert Mach number and temperature to arrays
    M, T_keV = np.broadcast_arrays(
        np.asarray(M, dtype=float),
        np.asarray(T_keV, dtype=float),
    )

    # Mach number -> electron spectral index
    s = s_of_M(M, gamma)

    # Temperature -> minimum dimensionless electron energy
    e_min_vals = emin_of_TkeV(T_keV)

    # Query the table at (s, log10(e_min))
    points = np.column_stack(
        (s.ravel(), np.log10(e_min_vals.ravel()))
    )
    log_ratio = interp_log_ratio(points)

    # Recover I_Psi / I_spec from its base-10 logarithm
    return np.power(10.0, log_ratio).reshape(s.shape)


# In[ ]:


def get_subhalo_maxM(Halo_ID, Sub_GrNr, Sub_Mass):

    """
    Finds the indices of the top three most massive subhalos for a given halo.

    Parameters:
    - Halo_ID: Array of halo IDs (single value or array with specific halo ID to match).
    - Sub_GrNr: Array indicating the group number (halo) each subhalo belongs to.
    - Sub_Mass: Array of subhalo masses.

    Returns:
    - Numpy array (CenterSub_Index, SecondSub_Index, ThirdSub_Index):
      - CenterSub_Index: Index of the most massive subhalo in the halo.
      - SecondSub_Index: Index of the second most massive subhalo.
      - ThirdSub_Index: Index of the third most massive subhalo.
    """

    find_Sub = np.where(Sub_GrNr == Halo_ID)[0]
    find_Sub_Mass = Sub_Mass[find_Sub]
    find_Sub_Mass_Sorted = np.argsort(find_Sub_Mass)

    CenterSub_Index = np.where(Sub_GrNr == Halo_ID)[0][find_Sub_Mass_Sorted[-1]]

    CenterSub_Index = np.array(CenterSub_Index)

    return CenterSub_Index#, SecondSub_Index, ThirdSub_Index

def Get_AvgSFR(SubhaloGrNr, SubhaloSFR, FOF_Halo_IDs):
    AvgSFR = np.zeros(FOF_Halo_IDs.shape)

    for i in range(len(FOF_Halo_IDs)):
        current_fof_id = FOF_Halo_IDs[i]
        sub_in_fof = (SubhaloGrNr == current_fof_id)
        subSFR_in_fof = SubhaloSFR[sub_in_fof]
        AvgSFR[i] = np.mean(subSFR_in_fof)

    return AvgSFR

def Get_HaloIDs(TargetHalo_cat, SubhaloMassDef):
    """
    Extracts halo IDs and their properties from the HDF5 catalog.

    Parameters:
    - TargetHalo_cat: Path to the HDF5 file containing the target halo catalog.

    Returns:
    - Target_Halo_IDs: List of selected halo IDs.
    - Subhalo_MaxMasses: Maximum subhalo masses for each selected halo.
    - Target_Halo_Rs_Crit200: Halo critical radius R_Crit200.
    - Target_GroupPoses: Positions of the selected halos.
    - Galaxy_nums: Number of subhalos per halo.
    """

    with h5py.File(TargetHalo_cat, 'r') as Target_hdf:
        # Read FOF Halo Info
        FOF_Halo_IDs =  Target_hdf['Group/FOF_Halo_IDs'][:]
        GroupFirstSub = Target_hdf['Group/GroupFirstSub'][:]

        Group_Nsubs = Target_hdf['Group/GroupNsubs'][:]

        # Read Subhalo Info
        SubhaloGrNr = Target_hdf['Subhalo/SubhaloGrNr'][:]
        SubhaloMass =  Target_hdf[f'Subhalo/{SubhaloMassDef}'][:]
        SubhaloSFR = Target_hdf['Subhalo/SubhaloSFR'][:]
        Subhalo_IDs = Target_hdf['Subhalo/Subhalo_IDs'][:]

    # Find FOF halos with subhalos and Read Info
    Indices_HaloWithSub = np.where( GroupFirstSub != -1)[0]
    Target_Halo_IDs = FOF_Halo_IDs[Indices_HaloWithSub]


    # Initialize array to get the central subhalo, second massive subhalo, and the third massive subhalo

    # Initialize galaxy numbers
    Galaxy_nums = Group_Nsubs[np.isin(FOF_Halo_IDs, Target_Halo_IDs)]
    AvgSFR = Get_AvgSFR(SubhaloGrNr, SubhaloSFR, FOF_Halo_IDs)

    # Initialize the Center subhalo info of each fof halo (consider it as the FOF halo itself using SubFind algorithms)
    Center_SubhaloIDs = np.zeros(Target_Halo_IDs.shape)
    Center_SubhaloSFR = np.zeros(Target_Halo_IDs.shape)

    for i in range(len(Target_Halo_IDs)):
        # locate the current FOF Halo ID
        Halo_ID = Target_Halo_IDs[i]

        # Find the indices of center, second, third subhalos
        CenterSub_Index = get_subhalo_maxM(Halo_ID, SubhaloGrNr, SubhaloMass)

        Center_SubhaloIDs[i] = Subhalo_IDs[CenterSub_Index]
        Center_SubhaloSFR[i] = SubhaloSFR[CenterSub_Index]

    return (Target_Halo_IDs, Galaxy_nums, AvgSFR,
            Center_SubhaloIDs, Center_SubhaloSFR)


# In[ ]:


import requests
import time

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key": "aef2c971d6020b2c432a5c8b0a60f7a8"}

def get(path, params=None, max_retries=5, retry_delay=2):
    for attempt in range(max_retries):
        try:
            r = requests.get(path, params=params, headers=headers, timeout=30)
            r.raise_for_status()  # 抛出非 200 错误
            if r.headers.get('content-type') == 'application/json':
                return r.json()
            if 'content-disposition' in r.headers:
                filename = r.headers['content-disposition'].split("filename=")[1]
                with open(filename, 'wb') as f:
                    f.write(r.content)
                return filename
            return r  # fallback
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

# Issue a request to the API root
r = get(baseUrl)

# Print out all the simulation names
names = [sim['name'] for sim in r['simulations']]
# Get the index of TNG300-1
i = names.index('TNG-Cluster')
# Get the info of simulation Illustris-3
sim = get( r['simulations'][i]['url'] )
sim.keys()

# get the snaps info this simulation
snaps = get(sim['snapshots'])

# Sim Box parameters
BoxSize = sim['boxsize'] # unit: ckpc/h
for item in snaps:
    if int(item['number']) == Snap_Index:
        snap_info = item
        break

Redshift = float(snap_info['redshift'])
z = Redshift
a = 1.0 / (1.0 + z)

print(f"Snapshot {Snap_Index}: z={z:.6f}, a={a:.6f}")


# In[ ]:


import pandas as pd
from pathlib import Path

project_dir = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)

cutout_dir = Path('/users/ckong13/data/Chuiyang/TNGCluster_Cutout/snap91')

cat_df = pd.read_csv(project_dir / 'fof_halo_to_sub91.csv')

id_columns = [
    'FOF_Halo_ID_At99',
    'Subhalo_ID_At91',
    'FOF_Halo_ID_At91',
]

# Stop if any target lacks a valid match.
if cat_df[id_columns].isna().any().any():
    raise ValueError("The matching table contains missing IDs.")

cat_df[id_columns] = cat_df[id_columns].astype('int64')

if (cat_df[id_columns] < 0).any().any():
    raise ValueError("The matching table contains negative IDs.")

if cat_df['FOF_Halo_ID_At99'].duplicated().any():
    raise ValueError("Multiple rows match the same snap99 FOF halo.")


# In[ ]:


def generate_radio(cutout_name):

    with h5py.File(cutout_name,"r") as f:
        #print(f['PartType0'].keys())
        Velocities = f['PartType0/Velocities'][:]
        Coordinates = f['PartType0/Coordinates'][:]
        Density = f['PartType0/Density'][:]
        ElectronAbundance = f['PartType0/ElectronAbundance'][:]
        InternalEnergy = f['PartType0/InternalEnergy'][:]
        Masses = f['PartType0/Masses'][:]
        Machnumber = f['PartType0/Machnumber'][:]
        MagneticField = f['PartType0/MagneticField'][:]
        EnergyDissipation = f['PartType0/EnergyDissipation'][:]

    # the hydrogen mass fraction
    XH = 0.76
    # the mean molecular weight
    mu = 4.0 / (1.0 + 3.0*XH + 4.0*XH* ElectronAbundance) *c.m_p.cgs.value

    # the adiabatic index gamma = 5/3
    # the Boltzmann constant c.k_B.cgs
    temperature = (5/3 - 1)* InternalEnergy/c.k_B.cgs.value * np.power(10,10) * mu

    T_keV = (k_B * (temperature * u.K)).to(u.keV).value

    ## However, as we are doing post processing, with Mach number and energy dissipation fields already given,
    ## we only need to find shock using Mach number and energy dissipation

    shock = (Machnumber > 1.3) & (EnergyDissipation >0)

    pos = Coordinates[shock]*a/h0 # physical units
    M   = Machnumber[shock]
    E   = EnergyDissipation[shock].astype(np.float64)/a*Econv *1e-44 # convert to E/(1e44 erg/s)
    B_uG   = vecnorm(MagneticField[shock]) * h0/a**2 * UnitB_G * 1e6 #convert unit of B to \muG
    T = T_keV[shock]
    s = s_of_M(M, gamma=GAMMA)
    Bcmb = Bcmb_uG(z)

    # Integral ratio: I_Psi / I_spec
    ratio = spectral_ratio_MT(M, T, gamma=GAMMA)

    spectral_factor = (
        np.power(SPECTRAL_BASE, 0.5 * s) * ratio
    )

    # Spectral radio power, in W/Hz
    w = (
        POWER_PREF_W_HZ
        * (XI_E / 0.05)
        * E
        * spectral_factor
        * np.power(B_uG, 1.0 + 0.5 * s)
        / (B_uG**2 + Bcmb**2)
        * np.power(NU_MHZ / NU_REF_MHZ, -0.5 * s)
    )

    return pos, w



# In[10]:


for row in cat_df.itertuples(index=False):
    fof99 = int(row.FOF_Halo_ID_At99)
    fof91 = int(row.FOF_Halo_ID_At91)
    sub91 = int(row.Subhalo_ID_At91)

    cutout_name = cutout_dir / f'cutout_sub{sub91}_FOF{fof91}.hdf5'

    pos, w = generate_radio(cutout_name)

    fname = project_dir / f'radio_FOF{fof91}_sub{sub91}.npz'

    np.savez_compressed(
        fname,
        pos=np.asarray(pos, dtype=np.float64),
        w=np.asarray(w, dtype=np.float64),
        snap=Snap_Index,
        redshift=z,
        fof99=fof99,
        fof91=fof91,
        sub91=sub91,
    )

    print(f'Finished: FOF99={fof99} → FOF91={fof91}, sub91={sub91}')

