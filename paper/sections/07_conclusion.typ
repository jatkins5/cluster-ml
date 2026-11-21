= Conclusion

We have developed a convolutional neural network (CNN) to predict the time since collision (TSC) of galaxy cluster mergers from simulated radio and X-ray images, achieving a correlation between true and predicted values of $r^2 = 0.511$ on the TNGCluster dataset.
This is a strong improvement over a baseline XGBoost model trained on hand-crafted features, which achieves $r^2 = 0.290$, and not far from an upper-bound Pooled CNN model using all projections, which achieves $r^2 = 0.544$.
It is also only slightly worse than the performance of a simple linear regression on the double-relic separation distance developed by @Lee2025, which achieves $r^2 = 0.69$ but is only applicable to the small fraction of clusters which host double relics.
Ablations show that the model is mainly using radio data for its predictions, with X-ray data providing a negligible boost in performance, and that most of its accuracy is from discriminating between recent and old mergers.
We hope to improve performance by adding more training data from other simulations, and by experimenting with pretrained vision foundation models and synthetic data generation techniques.
