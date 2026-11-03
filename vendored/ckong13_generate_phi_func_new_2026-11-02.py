#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Packages
import h5py
import numpy as np
from astropy import constants as c
from astropy import units as u
from scipy import special
import mpmath as mp
from astropy.constants import k_B, m_e
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


# In[2]:


# parameter
GAMMA = 5.0/3.0

# helper functions converting s to other variables

def r_of_s(s):
    s = np.asarray(s, dtype=float)
    return (s+2.0)/(s-1.0)

def q_of_s(s, gamma = GAMMA):
    s = np.asarray(s, dtype=float)
    r = r_of_s(s)
    A = (r*(gamma+1)-(gamma-1))/((gamma+1)-r*(gamma-1))

    return A*np.power(r, -gamma)


# In[ ]:


# Phi func

def Ipsi(s):
    s = np.asarray(s, dtype=float)
    logI = ( (s/2.0)*np.log(2.0)
            -np.log(s-1.0)-np.log(s+2.0)
            +special.gammaln(s/4.0 + 11.0/6.0)
            +special.gammaln(s/4.0 +  1.0/6.0))
    return np.exp(logI)

def I_spec(s, e_min, e_max=1e7):
    s = np.asarray(s, dtype=float)
    e_min = np.asarray(e_min, dtype=float)
    x = e_min / e_max
    a = 2.0 - s
    b = s - 1.0

    # mp.betainc(a,b, x, 1) = ∫_x^1 t^(a-1) (1-t)^(b-1) dt
    vec = np.vectorize(lambda ai, bi, xi: float(mp.betainc(ai, bi, xi, 1.0, regularized=False)))
    integral = vec(a, b, x)
    return (e_max**(2.0 - s)) * integral

def psi_func(s, e_min, e_max=1e7):
    """Return the spectral ratio I_Psi / I_spec."""
    return Ipsi(s) / I_spec(s, e_min, e_max)


# In[ ]:


# Get the interpolation table

s_min= 2.000
s_max= 8.0
Ns = 500
emin_min = 1e-10
emin_max = 30.0
Ne = 600
e_max = 1e7

s_grid = np.linspace(s_min, s_max, Ns)
loge_grid = np.linspace(np.log10(emin_min), np.log10(emin_max), Ne)

S, LOGE = np.meshgrid(s_grid, loge_grid, indexing="ij")

Emin = 10**LOGE

psi_grid = psi_func(S, Emin, e_max)


# In[ ]:


def r_of_M(M, gamma=GAMMA):
    M = np.asarray(M, dtype=float)
    return (gamma + 1) * M**2 / ((gamma - 1) * M**2 + 2)

def s_of_M(M, gamma=GAMMA):
    r = r_of_M(M, gamma)
    return (r + 2) / (r - 1)

# give range of Mach number
M = np.linspace(1.3, 20, 2000)
s = s_of_M(M)

plt.figure()
plt.plot(M, s)
plt.xlabel("Mach number M")
plt.ylabel("spectral index s")
plt.ylim(1.9, 8.1)
plt.grid(True, alpha=0.3)
plt.show()


# In[ ]:


# ---------- 1) determin e_min, Psi(s) ----------
emin_targets = [1e-10, 1e-6, 1e-3, 1e-1, 1.0, 3.0]
plt.figure()
for emin in emin_targets:
    j = np.argmin(np.abs(loge_grid - np.log10(emin)))
    plt.plot(s_grid, psi_grid[:, j], label=f"emin={10**loge_grid[j]:.1e}")
plt.xlabel("s")
plt.ylabel(r"$I_\Psi / I_{\rm spec}$")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- 2) determin s，Psi(emin) ----------
s_targets = [2.1, 2.5, 3.0, 4.0, 6.0, 8.0]
emin_vals = 10**loge_grid

plt.figure()
for s0 in s_targets:
    i = np.argmin(np.abs(s_grid - s0))
    plt.plot(emin_vals, psi_grid[i, :], label=f"s={s_grid[i]:.2f}")
plt.xlabel("emin")
plt.ylabel(r"$I_\Psi / I_{\rm spec}$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- 3) 2D heatmap: Psi(s, log10(emin)) ----------
plt.figure()
im = plt.imshow(
    np.log10(psi_grid),
    origin="lower",
    aspect="auto",
    extent=[loge_grid.min(), loge_grid.max(), s_grid.min(), s_grid.max()],
)
plt.xlabel("log10(emin)")
plt.ylabel("s")
plt.colorbar(im, label=r"$\log_{10}(I_\Psi / I_{\rm spec})$")
plt.tight_layout()
plt.show()


# In[ ]:


table_path = (
    "./radio_spectral_ratio_table.npz"
)

np.savez_compressed(
    table_path,
    s_grid=s_grid,
    loge_grid=loge_grid,
    log10_spectral_ratio=np.log10(psi_grid),
    gamma=GAMMA,
    e_max=e_max,
    schema_version=1,
)

print(f"Saved to {table_path}")

