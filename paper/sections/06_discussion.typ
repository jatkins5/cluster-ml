= Discussion and Future Work
 
@Lee2025 found that, in TNGCluster and TNG300-1 simulations, the TSC is closely correlated with the separation distance between double radio relics, with $r^2 = 0.69$ (for single relics, a somewhat weaker performance is possible using the distance between the relic and the BCG, giving $r^2 = 0.53$).
Our CNN is not able to match this performance, with the Pooled CNN achieving $r^2 = 0.544$, upper bounding the performance of a single projection.
This gap may partially be due to the relative rarity of relic systems in the TNGCluster dataset; while we have not counted explicitly, only 5-10% of clusters in real datasets host relics, and our TNGCluster should reflect that distribution.
Non-relic clusters may have less predictable TSC values.

These conclusions are difficult to verify, however, since we only have ~350 cluster samples; at 5% relic fraction, that would only give us ~17 relic clusters to evaluate on, which is too small of a sample size to draw strong conclusions.
We are hoping to improve the sample size by adding more datasets, such as TNG300-1 and CAMELS.
This may also boost the salience of the geometric features of relics, which may be more predictive of TSC than the overall radio morphology.
We are also experimenting with adding hand-crafted features in addition to the radio and X-ray images to reflect existing measures of cluster merger activity, such as the BIC score, elongation ratio, and double- and single-relic distances.
Pretrained vision foundation models for astrophysics (Lastufka 2025) may also be a promising avenue to improve performance. 
However, none have been trained on diffuse cluster emission specifically, so it is unclear how much they will improve performance compared to a CNN trained from scratch on the TNGCluster dataset.
Finally, we would like to experiment with synthetic data generation techniques (@Caro2025; @Hsu2024; @Benyas2024) to augment our training data, which may help to improve performance by providing more examples of rare cluster types such as relic-hosting clusters.
Once we are confident model performance is sufficient, we will apply our model to the LoVoCCS dataset to predict TSC values for real clusters, and compare these predictions to existing measures of cluster merger activity such as the BIC score and the presence of radio relics.
