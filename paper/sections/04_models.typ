= Models and Cross-validation Protocol

To train all models, we use a 5-fold GroupKFold split on the 352 TNGCluster clusters, with the three projections of each cluster grouped together to prevent data leakage.
We report the $r^2$ correlation between the predicted and true TSC values on the held-out test set for each fold, as well as the mean absolute error (MAE) and root mean squared error (RMSE) for each TSC definition.

== Models

*XGBoost Baseline*:
To establish a baseline for TSC prediction, we train an XGBoost regression model on the 17 handcrafted features extracted from TNGCluster data in @Kong2025: per-projection means and standard deviations of galaxy positions ($mu_(r,0)$, $mu_(r,1)$, $sigma_r$, $sigma_(r,0)$, $sigma_(r,1)$) and line-of-sight-corrected velocities ($mu_(v,0)$, $mu_(v,1)$, $sigma_v$, $sigma_(v,0)$, $sigma_(v,1)$); subhalo counts $n_0$, $n_1$ assigned to each of two phase-space components; BIC scores favoring one- vs. two-component Gaussian mixtures in projected phase space ($"BIC"_1$, $"BIC"_2$); the elongation ratio of the X-ray surface-brightness isophotes; and two cluster-level features, $R_"500c"$ and the binary mass ratio.

*Shallow CNN*:
To leverage the spatial information in the radio and X-ray images, we train a shallow convolutional neural network (CNN) with four convolution blocks producing 32, 64, 128, and 256 output channels, followed by a global average pooling layer and a two-layer fully-connected MLP.
Each block consists of two $3 times 3$ convolutions, each followed by batch normalization and ReLU, and ends with a $2 times 2$ max-pool that halves the spatial resolution.
We use the AdamW optimizer with a cosine learning rate schedule, and train on a $delta$-tunable Huber loss to mitigate the impact of outliers in TSC values:
$ L_delta (r) = cases(1/2 r^2 "if" |r| <= delta, delta (|r| - 1/2 delta) "otherwise"), $
where $r = hat(tau) - tau$ is the prediction residual. Huber loss is quadratic for small residuals, which preserves sensitivity near the bulk of the distribution, and linear for large residuals, which downweights outliers. Unless otherwise noted, we use $delta = 0.5$ Gyr.

*Pooled CNN*:
To combine the three orthogonal projections of each cluster and find an upper bound on predictive performance, we train a pooled CNN which passes each projection through a single shared encoder (identical in architecture to the Shallow CNN backbone, without the MLP head) and reduces the resulting three 256-dimensional feature vectors to one by element-wise mean across projections, before passing the pooled vector to a two-layer MLP head.

*Dual-encoder*:
To combine the radio and X-ray data, we train a dual-encoder architecture which trains separate CNN models for each modality, then concatenates the resulting feature maps before the fully-connected layers.
