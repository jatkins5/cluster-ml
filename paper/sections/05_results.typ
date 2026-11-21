= Results

#figure(
  table(
    columns: 5,
    [*Model*], [*Label*], [*$r^2$*], [*MAE (Gyr)*], [*RMSE (Gyr)*],
    [XGBoost], [`label_score`], [0.290], [0.469], [0.632],
    [Shallow CNN (radio)], [`label_score`], [0.472], [0.414], [0.544],
    [Shallow CNN (radio)], [`pseudo-TSC`], [0.529], [0.587], [0.806],
    [Shallow CNN (radio)], [`merger-TSC`], [0.511], [0.798], [1.073],
    [Pooled CNN (radio)], [`merger-TSC` ($delta=2$, 120 ep.)], [0.544], [0.778], [1.047],
    [Shallow CNN (X-ray)], [`merger-TSC`], [0.321], [0.938], [1.265],
    [Dual-encoder (radio + X-ray)], [`merger-TSC`], [0.511], [0.777], [1.073],
  ),
  caption: [Predictive performance of all models on the TNGCluster test set, averaged across 5 GroupKFold splits.]
) <results-table>

@results-table shows the performance of each choice of model architecture and TSC definition on the TNGCluster test set, averaged across the 5 GroupKFold splits.
As expected, the Pooled CNN has the best performance, with an $r^2$ of 0.544, due to its ability to leverage all three projections of a cluster.
However, using a single projection with the Shallow CNN still achieves $r^2$ values of 0.529 (for the `pseudo-TSC`) and 0.511 (for the `merger-TSC`), which are significant improvements over the XGBoost baseline's $r^2$ of 0.290.

The X-ray data, on the other hand, does not seem to provide much information; its Shallow CNN version only achieves an $r^2$ of 0.321, and the Dual-encoder model which combines the radio and X-ray data does not improve over the radio-only Shallow CNN, with an $r^2$ of 0.511.
Since X-ray emission mostly traces the thermal gas in the cluster, while radio emission traces the non-thermal relativistic electrons accelerated by shocks during mergers, this suggests that the non-thermal processes captured by the radio data are more informative of merger history than the thermal processes captured by the X-ray data.

Note that the above results are all for 128$times$128 resolution images; we also ran 256$times$256 resolution simulations, which achieved similar performance at $4 times R_"500c"$ field of view, but worse performance at $2 times R_"500c"$ field of view, with $r^2 = 0.480$.
Thus, accuracy is not strongly determined by resolution, compared to field of view.

We also experimented with training a model on only recent mergers, with $tau < 2$ Gyr, and sample size of 258.
This model performed very poorly, with $r^2 = 0.149$.
Though the sample size is smaller compared to the full dataset ($n=352$), we posit that the difference is mainly due to the model being sensitive only to the difference between recent and old mergers.
@tsc-true-vs-pred displays the true vs. predicted `merger-TSC` values for the Shallow and Pooled CNN models, which shows that the model is able to capture the overall trend of increasing TSC, but struggles to predict the exact TSC values, especially for recent mergers with $tau < 2$ Gyr.

#figure(
  image("../figures/tsc_true_vs_pred.png", width: 100%),
) <tsc-true-vs-pred>
