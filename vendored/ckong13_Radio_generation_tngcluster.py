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


# In[2]:


# Cosmic parameters
z = 0.
a = 1/(1+z)
from astropy.cosmology import Planck15
h0 = Planck15.h


# In[3]:


# unit conversion

# compute the numerical factor in cgs from the units of B in TNG-Cluster
expr = np.sqrt(1e10 * u.Msun / u.kpc) * (u.km/u.s) / u.kpc

# convert the expression to sqrt(dyne/cm^2) (i.e. sqrt(pressure) in cgs)
# since 1 dyne/cm^2 = 1 g/(cm s^2)
expr_cgs = expr.to(u.g**0.5 / (u.cm**0.5 * u.s))   # same as sqrt(dyne/cm^2)

UnitB_G = expr_cgs.value 

Econv = ((1e10 * u.Msun / u.kpc) * (u.km/u.s)**3).to(u.erg/u.s).value


# In[4]:


# define the psi function using interpolation

data = np.load("normalized_psi_table.npz")
s_grid = data["s_grid"]
loge_grid = data["loge_grid"]     # log10(emin)
normalized_psi_grid = data["normalized_psi_grid"]

interp_psi = RegularGridInterpolator(
    (s_grid, loge_grid),
    normalized_psi_grid,
    bounds_error=False, 
    fill_value=None
)


# In[5]:


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

def phi_M(M, T_keV, gamma=GAMMA):
    s = s_of_M(M, gamma)
    e_min_vals = emin_of_TkeV(T_keV)
    log_e_min_vals = np.log10(e_min_vals)

    pts = np.column_stack([s, log_e_min_vals])
    phi = interp_psi(pts)
    return phi


# In[6]:


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
    #SecondSub_Index = np.where(Sub_GrNr == Halo_ID)[0][find_Sub_Mass_Sorted[-2]]
    #ThirdSub_Index = np.where(Sub_GrNr == Halo_ID)[0][find_Sub_Mass_Sorted[-3]]

    CenterSub_Index = np.array(CenterSub_Index)
    #SecondSub_Index = np.array(SecondSub_Index)
    #ThirdSub_Index = np.array(ThirdSub_Index)
    
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
        #GroupPos = Target_hdf['Group/GroupPos'][:]
        #Group_R_Crit200 = Target_hdf['Group/Group_R_Crit200'][:]
        Group_Nsubs = Target_hdf['Group/GroupNsubs'][:]

        # Read Subhalo Info
        SubhaloGrNr = Target_hdf['Subhalo/SubhaloGrNr'][:]
        SubhaloMass =  Target_hdf[f'Subhalo/{SubhaloMassDef}'][:]
        #SubhaloSpin = Target_hdf['Subhalo/SubhaloSpin'][:]
        #SubhaloVelDisp = Target_hdf['Subhalo/SubhaloVelDisp'][:]
        SubhaloSFR = Target_hdf['Subhalo/SubhaloSFR'][:]
        Subhalo_IDs = Target_hdf['Subhalo/Subhalo_IDs'][:]
        #Subhalo_MassRad = 2*Target_hdf['Subhalo/SubhaloHalfmassRad'][:]

    # Find FOF halos with subhalos and Read Info
    Indices_HaloWithSub = np.where( GroupFirstSub != -1)[0]
    Target_Halo_IDs = FOF_Halo_IDs[Indices_HaloWithSub]
    #Target_Halo_Rs_Crit200 = Group_R_Crit200[Indices_HaloWithSub]
    #Target_GroupPoses = GroupPos[Indices_HaloWithSub]

    # Initialize array to get the central subhalo, second massive subhalo, and the third massive subhalo

    # Initialize galaxy numbers
    Galaxy_nums = Group_Nsubs[np.isin(FOF_Halo_IDs, Target_Halo_IDs)]
    AvgSFR = Get_AvgSFR(SubhaloGrNr, SubhaloSFR, FOF_Halo_IDs)

    # Initialize the Center subhalo info of each fof halo (consider it as the FOF halo itself using SubFind algorithms)
    Center_SubhaloIDs = np.zeros(Target_Halo_IDs.shape)
    #Center_SubhaloMasses = np.zeros(Target_Halo_IDs.shape)
    #Center_SubhaloVelDisp = np.zeros(Target_Halo_IDs.shape)
    Center_SubhaloSFR = np.zeros(Target_Halo_IDs.shape)
    #Center_SubhaloSpin = np.zeros((Target_Halo_IDs.shape[0],3))
    #Center_SubhaloMassRad = np.zeros(Target_Halo_IDs.shape)
      
    # Initialize the Second Subhalo info
    #Second_SubhaloIDs = np.zeros(Target_Halo_IDs.shape)
    #Second_SubhaloMasses = np.zeros(Target_Halo_IDs.shape)
    #Second_SubhaloVelDisp = np.zeros(Target_Halo_IDs.shape)
    #Second_SubhaloSFR = np.zeros(Target_Halo_IDs.shape)
    #Second_SubhaloSpin = np.zeros((Target_Halo_IDs.shape[0],3))
    #Second_SubhaloMassRad = np.zeros(Target_Halo_IDs.shape)

    # Initialize the Third Subhalo info
    #Third_SubhaloIDs = np.zeros(Target_Halo_IDs.shape)
    #hird_SubhaloMasses = np.zeros(Target_Halo_IDs.shape)
    #Third_SubhaloVelDisp = np.zeros(Target_Halo_IDs.shape)
    #Third_SubhaloSFR = np.zeros(Target_Halo_IDs.shape)
    #Third_SubhaloSpin = np.zeros((Target_Halo_IDs.shape[0],3))
    #Third_SubhaloMassRad = np.zeros(Target_Halo_IDs.shape)

    for i in range(len(Target_Halo_IDs)):
        # locate the current FOF Halo ID
        Halo_ID = Target_Halo_IDs[i]

        # Find the indices of center, second, third subhalos
        CenterSub_Index = get_subhalo_maxM(Halo_ID, SubhaloGrNr, SubhaloMass)

        Center_SubhaloIDs[i] = Subhalo_IDs[CenterSub_Index]
        #Center_SubhaloMasses[i] = SubhaloMass[CenterSub_Index]
        #Center_SubhaloVelDisp[i] = SubhaloVelDisp[CenterSub_Index]
        Center_SubhaloSFR[i] = SubhaloSFR[CenterSub_Index]
        #Center_SubhaloSpin[i] = SubhaloSpin[CenterSub_Index]
        #Center_SubhaloMassRad[i] = Subhalo_MassRad[CenterSub_Index]

        #Second_SubhaloIDs[i] = Subhalo_IDs[SecondSub_Index]
        #Second_SubhaloMasses[i] = SubhaloMass[SecondSub_Index]
        #Second_SubhaloVelDisp[i] = SubhaloVelDisp[SecondSub_Index]
        #Second_SubhaloSFR[i] = SubhaloSFR[SecondSub_Index]
        #Second_SubhaloSpin[i] = SubhaloSpin[SecondSub_Index]
        #Second_SubhaloMassRad[i] = Subhalo_MassRad[SecondSub_Index]

        #Third_SubhaloIDs[i] = Subhalo_IDs[ThirdSub_Index]
        #Third_SubhaloMasses[i] = SubhaloMass[ThirdSub_Index]
        #Third_SubhaloVelDisp[i] = SubhaloVelDisp[ThirdSub_Index]
        #Third_SubhaloSFR[i] = SubhaloSFR[ThirdSub_Index]
        #Third_SubhaloSpin[i] = SubhaloSpin[ThirdSub_Index]
        #Third_SubhaloMassRad[i] = Subhalo_MassRad[ThirdSub_Index]

    return (Target_Halo_IDs, Galaxy_nums, AvgSFR,
            Center_SubhaloIDs, Center_SubhaloSFR)
    '''
    return (Target_Halo_IDs, Galaxy_nums, Target_Halo_Rs_Crit200, Target_GroupPoses, AvgSFR,
            Center_SubhaloIDs, Center_SubhaloMasses, Center_SubhaloVelDisp, Center_SubhaloSFR, Center_SubhaloSpin, Center_SubhaloMassRad,
            Second_SubhaloIDs, Second_SubhaloMasses, Second_SubhaloVelDisp, Second_SubhaloSFR, Second_SubhaloSpin, Second_SubhaloMassRad,
            Third_SubhaloIDs, Third_SubhaloMasses, Third_SubhaloVelDisp, Third_SubhaloSFR, Third_SubhaloSpin, Third_SubhaloMassRad
            )
        '''


# In[7]:


import requests
import time

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key": "REDACTED-TNG-API-KEY"}

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
Snap_Index = 99 # the snapshots index in the total 100 snapshots taking at different z
BoxSize = sim['boxsize'] # unit: ckpc/h
Redshift = snaps[Snap_Index]['redshift'] # current redshift of our current snap


# In[8]:


cat_name = '/users/ckong13/data/Chuiyang/targethalo_cat_TNGCluster/targethalo_cat_099/TargetHalo_MergerCat_099.hdf5'
results = Get_HaloIDs(cat_name, SubhaloMassDef='SubhaloMass')


# In[9]:


with h5py.File('./TNG-Cluster_Catalog.hdf5', 'r') as f:
    origID_parentsim = f['origID'][:]
    haloID__TNGCluster = f['haloID'][:] # target halo id at redshift 0


# In[10]:


(Target_Halo_IDs, Galaxy_nums, AvgSFR,
Center_SubhaloIDs, Center_SubhaloSFR) = results

Subhalo_IDs = Center_SubhaloIDs[np.isin(Target_Halo_IDs, haloID__TNGCluster)]
Subhalo_IDs = Subhalo_IDs.astype(int)


# In[11]:


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

    phi = phi_M(M, T, gamma =GAMMA)

    w = 5.2e+23 * E* (B_uG**(1.0 + 0.5*s)) / (B_uG**2 + Bcmb**2) * phi
    
    return pos, w
    


# In[12]:


for i in range(len(haloID__TNGCluster)):
    current_fof_id = haloID__TNGCluster[i]
    current_sub_id = Subhalo_IDs[i]
    
    cutout_name = f"/users/ckong13/data/Chuiyang/TNGCluster_Cutout/snap99/cutout_sub{current_sub_id}_FOF{current_fof_id}.hdf5"
    
    pos, w = generate_radio(cutout_name)
    
    fname = f'./radio_FOF{current_fof_id}_sub{current_sub_id}.npz'
    
    np.savez_compressed(
        fname,
        pos=np.asarray(pos, dtype=np.float64),
        w=np.asarray(w, dtype=np.float64),
    )
    
    print(f'finish FOF {i}')

