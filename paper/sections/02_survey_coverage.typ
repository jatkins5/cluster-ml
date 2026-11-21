= Observational Survey Coverage of LoVoCCS

LoVoCCS (the Local Volume-Complete Cluster Survey; @Fu2022) is a program to observe nearly every low-redshift ($0.03 < z < 0.12$) X-ray-luminous galaxy cluster with the Dark Energy Camera (DECam) in the u, g, r, i, and z bands.
Being volume-complete, LoVoCCS is designed to provide a comprehensive census of galaxy clusters in the local universe, enabling studies of cluster properties, galaxy evolution, and cosmology which are not biased by selection effects that can arise in flux-limited surveys.
As such, LoVoCCS observations are an unbiased test case for machine learning models trained on simulated cluster data.

For this study, we use 106 LoVoCCS clusters which have already been observed, and cross-match to other surveys to obtain multi-spectrum observations of these clusters.
We use a 5' search radius around each cluster center against VizieR-hosted catalogs and dedicated archives (10' for LoTSS, matched in bulk against the locally-downloaded PyBDSF source catalog rather than via VizieR).

== Radio Coverage

*VLASS Epoch 1*:
The VLA Sky Survey (VLASS; @Lacy2020) is a radio survey conducted with the Karl G. Jansky Very Large Array (VLA) at S-band (2-4 GHz), with a 2.5" resolution.
We find 76 matching LoVoCCS clusters with VLASS Epoch 1 coverage, with the misses generally being at declinations below $-40degree$ which are not covered by VLASS.

*LoTSS DR3*: The LOFAR Two-metre Sky Survey (LoTSS) Data Release 3 @Shimwell2026 is a radio survey conducted with the Low-Frequency Array (LOFAR) at 120-168 MHz, with a 6" resolution (9" below declination $+10degree$), covering 88% of the northern sky.
We find 26 matching LoVoCCS clusters in LoTSS DR3, with the misses generally being at southern declinations not covered by the LOFAR survey.

*Parkes*:
The Parkes radio telescope has been used in several surveys of the southern sky; we cross-match against the Parkes-MIT-NRAO (PMN) 4.85 GHz survey (@Griffith1993; @Wright1994) and the multi-frequency Parkes Radio Sources Catalogue (PKSCAT90, 80 MHz–5 GHz; @Otrupcek1991) via VizieR.
We find 28 matching LoVoCCS clusters across the two catalogs (28 in PMN, 12 in PKSCAT90, with PKSCAT90 matches a subset of PMN).

*MeerKAT MGCLS DR1*:
The MeerKAT Galaxy Cluster Legacy Survey (MGCLS) is a radio survey conducted with the MeerKAT telescope at L-band (0.9-1.67 GHz), with a 8" resolution.
We find 18 matching LoVoCCS clusters with MGCLS coverage.

== X-ray Coverage

*eROSITA eRASS1*:
The extended ROentgen Survey with an Imaging Telescope Array (eROSITA) is an X-ray survey conducted with the eROSITA instrument on the Spektr-RG satellite, covering the 0.2-10 keV energy range.
While eROSITA is nominally an all-sky survey, the first eROSITA All-Sky Survey (eRASS1) has only been publicly released for the western Galactic hemisphere ($359.94degree > l > 179.94degree$) by the eROSITA-DE consortium @Merloni2024; the eastern half is held by the Russian consortium and remains proprietary.
We find 59 matching LoVoCCS clusters with eRASS1 coverage.

