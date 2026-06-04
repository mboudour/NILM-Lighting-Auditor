import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import json
import time

print("Running Clustering Sensitivity (k=2,3,4,5)...")
latent_reps = np.load("/home/ubuntu/computations_run/computations/outputs/stage1/latent_representations.npy")

# Stratified subsample
np.random.seed(42)
indices = np.random.choice(len(latent_reps), size=5000, replace=False)
subsample = latent_reps[indices]

results = {}
for k in [2, 3, 4, 5]:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init=10)
    labels = kmeans.fit_predict(subsample)
    score = silhouette_score(subsample, labels)
    results[k] = score
    print(f"k={k}: Silhouette Score = {score:.4f}")

print("\nRunning Statistical Significance (Bootstrap for k=3)...")
bootstrap_scores = []
for i in range(30): # 30 iterations for a quick CI
    boot_indices = np.random.choice(len(latent_reps), size=5000, replace=True)
    boot_sample = latent_reps[boot_indices]
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=i, batch_size=1024, n_init=10)
    labels = kmeans.fit_predict(boot_sample)
    score = silhouette_score(boot_sample, labels)
    bootstrap_scores.append(score)

mean_score = np.mean(bootstrap_scores)
std_score = np.std(bootstrap_scores)
ci_lower = mean_score - 1.96 * std_score
ci_upper = mean_score + 1.96 * std_score
print(f"Bootstrap Mean: {mean_score:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

