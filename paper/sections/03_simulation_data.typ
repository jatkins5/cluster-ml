= Simulation Training Set

To build the training dataset, we produce mock radio and X-ray observations of galaxy clusters from the TNGCluster cosmological simulation, and label them with their merger history using the time since collision (TSC) metric.
We produce HDF5 files with 128$times$128 or 256$times$256 resolution, and apply an $"arcsinh"$ transformation per pixel to compress the large dynamic range of the radio weights while preserving low-intensity structure: $x' = "arcsinh"(x)$.
Inputs are subsequently standardized per fold (mean and standard deviation computed on the training partition) before being passed to the CNN.

== TNGCluster radio data

The TNGCluster dataset is a collection of simulated radio observations of galaxy clusters from the IllustrisTNG cosmological simulation.
We extract 352 clusters at snapshot 99 ($z=0$) at a $4 times R_"500c"$ field of view into `.npz` files, where $R_"500c"$ denotes the radius enclosing a mean overdensity 500 times the critical density of the universe.
Each file contains shock particle positions and radio emission weights, computed according to the shock-acceleration model of @Hoeft2007:

$ (d P (nu_i))/(d nu_i) = 5.2 times 10^(23) times E_i (B_i^(1+s_i/2))/(B_i^2 + B_"CMB"^2) nu_i Psi(M) $

where $E_i$ is the shock energy dissipation rate, $B_i$ is the magnetic field strength, $s_i$ is the shock spectral index, $B_"CMB"$ is the equivalent magnetic field strength of the cosmic microwave background, and $Psi(M)$ contains the dependence on the Mach number $M$ of the shock.
@Lee2025 use a slightly different equation here, using a $Phi(M)$ function that is not defined; we use $Psi(M)$ because it is defined explicitly in @Hoeft2007, and we do not think it should make a significant difference to the resulting radio emission maps.
We use this method to generate three orthogonal projection maps of each cluster.
In the pooled methods below, these will be treated as a single sample, but generally we can treat each projection as independent.

== X-ray data

To simulate X-ray data, we use the `pyXSIM` and `SOXS` packages to generate Chandra Advanced CCD Imaging Spectrometer I (ACIS-I) mock observations of the TNGCluster dataset.
As with the radio data, we generate three orthogonal projections of each cluster at a field of view of $2 times R_"500c"$, and extract 0.1-2.0 keV images (the event files contain up to 12 keV photons, but we filter for the soft X-ray band to focus on the ICM emission and reduce noise from high-energy photons).

== Time Since Collision

The metric we use to quantify merger history is the time since collision (TSC, $tau$), which is the time elapsed since the most recent major merger event in the cluster's history.
We use three different definitions of TSC to understand which is most predictable from the radio and X-ray data:

#table(
  columns: 3,
  [*Label*], [*Source*], [*Definition*],
  [`label_score`], [@Kong2025], [Soft merger activity score within $plus.minus 1$ Gyr],
  [`pseudo-TSC`], [Interpolated from `label_score` curve], [$tau$ where `label_score` first crosses 0.5],
  [`merger-TSC`], [TSC merger catalog], [Cosmic-time difference since most recent collision snapshot],
)
