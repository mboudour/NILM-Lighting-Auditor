"""
Plot: Silhouette Score vs. Number of Clusters (k sensitivity)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

k_values = [2, 3, 4, 5]
scores = [0.3613, 0.3678, 0.3456, 0.3045]

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#4a90d9' if k != 3 else '#e07b39' for k in k_values]
bars = ax.bar(k_values, scores, color=colors, edgecolor='black', linewidth=0.8, width=0.5)

ax.set_xlabel('Number of Clusters (k)', fontsize=12)
ax.set_ylabel('Silhouette Score', fontsize=12)
ax.set_title('Clustering Sensitivity: Silhouette Score vs. k', fontsize=13)
ax.set_xticks(k_values)
ax.set_ylim(0.25, 0.40)
ax.axhline(y=max(scores), color='#e07b39', linestyle='--', linewidth=1.2, alpha=0.7)

for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{score:.4f}', ha='center', va='bottom', fontsize=10)

ax.annotate('Optimal k=3', xy=(3, 0.3678), xytext=(3.5, 0.375),
            arrowprops=dict(arrowstyle='->', color='#e07b39'),
            fontsize=10, color='#e07b39')

plt.tight_layout()
plt.savefig('/home/ubuntu/computations_run/computations/outputs/plots/clustering_sensitivity.png',
            dpi=150, bbox_inches='tight')
print("Saved clustering_sensitivity.png")
