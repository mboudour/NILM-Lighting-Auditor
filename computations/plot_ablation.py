"""
Plot: Ablation Study - Reconstruction MSE for different N-BEATS configurations
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

labels = ['Full N-BEATS\n(Trend + Seasonality)', 'No Trend Block\n(Seasonality only)', 'No Seasonality Block\n(Trend only)']
mse_values = [0.0218, 0.0319, 0.0353]
colors = ['#5ba85a', '#e07b39', '#c0392b']

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, mse_values, color=colors, edgecolor='black', linewidth=0.8, width=0.5)

ax.set_ylabel('Reconstruction MSE (lower is better)', fontsize=11)
ax.set_title('Ablation Study: Contribution of N-BEATS Blocks', fontsize=13)
ax.set_ylim(0, 0.045)

for bar, val in zip(bars, mse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.annotate('Best: both blocks\nwork together', xy=(0, 0.0218), xytext=(0.6, 0.035),
            arrowprops=dict(arrowstyle='->', color='#5ba85a'),
            fontsize=9, color='#5ba85a')

plt.tight_layout()
plt.savefig('/home/ubuntu/computations_run/computations/outputs/plots/ablation_study.png',
            dpi=150, bbox_inches='tight')
print("Saved ablation_study.png")
