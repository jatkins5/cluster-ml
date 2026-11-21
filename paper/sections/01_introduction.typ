= Introduction

Galaxy clusters are critical tracers of structure formation in the universe, and their merger history provides important insights into the growth of cosmic structure and the physics of the intracluster medium (ICM).
However, our understanding of mergers is hampered by weak observational constraints on merger timescales, which are typically estimated using indirect measures such as the presence of radio relics or the BIC score (a measure of X-ray morphology).
Recovering the time since collision (TSC) is a long-standing inverse problem.

In this work, we develop a convolutional neural network (CNN) model to predict the TSC of galaxy cluster mergers from simulated radio and X-ray cluster observations, generated from the TNGCluster dataset at $z=0$.
We also conduct a multi-wavelength census of the LoVoCCS (Local Volume-Complete Cluster Survey) sample to construct a real-world test set for our model across radio (VLASS, LoTSS, Parkes, MeerKAT) and X-ray (eROSITA) wavelengths.
