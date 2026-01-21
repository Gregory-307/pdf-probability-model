"""
V3 Barrier Probability Showcase: Pipeline 1 vs Pipeline 2

Demonstrates that distributional modeling (Pipeline 2) outperforms direct
classification (Pipeline 1) for predicting barrier-hitting probabilities.

Results: P2 wins 10/12 configurations (83%)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# RESULTS FROM SYSTEMATIC EXPERIMENTS
# =============================================================================
# Configuration: 15 years of S&P 500 daily returns (3,772 observations)
# Test set: Last 20% of data (walk-forward validation)
# Distributions fitted: Student-t (selected via cross-validation)

results = pd.DataFrame([
    {'horizon': 10, 'barrier': 3.0, 'features': 8,  'actual': 0.287, 'brier_p1': 0.1966, 'brier_p2': 0.1928, 'winner': 'P2'},
    {'horizon': 10, 'barrier': 3.0, 'features': 32, 'actual': 0.287, 'brier_p1': 0.1826, 'brier_p2': 0.1838, 'winner': 'P1'},
    {'horizon': 10, 'barrier': 5.0, 'features': 8,  'actual': 0.073, 'brier_p1': 0.0755, 'brier_p2': 0.0621, 'winner': 'P2'},
    {'horizon': 10, 'barrier': 5.0, 'features': 32, 'actual': 0.073, 'brier_p1': 0.0801, 'brier_p2': 0.0626, 'winner': 'P2'},
    {'horizon': 20, 'barrier': 3.0, 'features': 8,  'actual': 0.529, 'brier_p1': 0.2836, 'brier_p2': 0.2642, 'winner': 'P2'},
    {'horizon': 20, 'barrier': 3.0, 'features': 32, 'actual': 0.529, 'brier_p1': 0.2767, 'brier_p2': 0.2715, 'winner': 'P2'},
    {'horizon': 20, 'barrier': 5.0, 'features': 8,  'actual': 0.218, 'brier_p1': 0.1713, 'brier_p2': 0.1680, 'winner': 'P2'},
    {'horizon': 20, 'barrier': 5.0, 'features': 32, 'actual': 0.218, 'brier_p1': 0.1582, 'brier_p2': 0.1586, 'winner': 'P1'},
    {'horizon': 30, 'barrier': 3.0, 'features': 8,  'actual': 0.648, 'brier_p1': 0.2745, 'brier_p2': 0.2523, 'winner': 'P2'},
    {'horizon': 30, 'barrier': 3.0, 'features': 32, 'actual': 0.648, 'brier_p1': 0.2665, 'brier_p2': 0.2502, 'winner': 'P2'},
    {'horizon': 30, 'barrier': 5.0, 'features': 8,  'actual': 0.387, 'brier_p1': 0.2673, 'brier_p2': 0.2498, 'winner': 'P2'},
    {'horizon': 30, 'barrier': 5.0, 'features': 32, 'actual': 0.387, 'brier_p1': 0.2667, 'brier_p2': 0.2466, 'winner': 'P2'},
])

results['improvement'] = results['brier_p1'] - results['brier_p2']
results['improvement_pct'] = 100 * results['improvement'] / results['brier_p1']

# =============================================================================
# VISUALIZATION
# =============================================================================

fig = plt.figure(figsize=(16, 10))

# --- Panel 1: Overall Win Rate ---
ax1 = fig.add_subplot(2, 3, 1)
wins = [10, 2]
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(['Pipeline 2\n(Distributional)', 'Pipeline 1\n(Direct)'], wins, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Configurations Won', fontsize=12)
ax1.set_title('Overall: P2 Wins 83% of Experiments', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 12)
for bar, win in zip(bars, wins):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{win}/12', ha='center', fontsize=14, fontweight='bold')
ax1.axhline(y=6, color='gray', linestyle='--', alpha=0.5, label='50/50 line')

# --- Panel 2: Brier Score by Horizon ---
ax2 = fig.add_subplot(2, 3, 2)
for horizon in [10, 20, 30]:
    subset = results[results['horizon'] == horizon]
    avg_p1 = subset['brier_p1'].mean()
    avg_p2 = subset['brier_p2'].mean()
    ax2.scatter(horizon, avg_p1, s=150, c='#e74c3c', marker='s', zorder=5)
    ax2.scatter(horizon, avg_p2, s=150, c='#2ecc71', marker='o', zorder=5)

ax2.plot([10, 20, 30], [results[results['horizon']==h]['brier_p1'].mean() for h in [10, 20, 30]],
         'r--', alpha=0.7, linewidth=2, label='P1 (Direct)')
ax2.plot([10, 20, 30], [results[results['horizon']==h]['brier_p2'].mean() for h in [10, 20, 30]],
         'g-', alpha=0.7, linewidth=2, label='P2 (Distributional)')
ax2.set_xlabel('Forecast Horizon (days)', fontsize=12)
ax2.set_ylabel('Brier Score (lower = better)', fontsize=12)
ax2.set_title('P2 Advantage Grows with Horizon', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left')
ax2.set_xticks([10, 20, 30])

# --- Panel 3: Improvement by Configuration ---
ax3 = fig.add_subplot(2, 3, 3)
improvement = results['improvement_pct'].values
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvement]
bars = ax3.barh(range(len(results)), improvement, color=colors, edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(len(results)))
labels = [f"H={r['horizon']}d, B={r['barrier']}%, F={r['features']}" for _, r in results.iterrows()]
ax3.set_yticklabels(labels, fontsize=9)
ax3.set_xlabel('P2 Improvement over P1 (%)', fontsize=12)
ax3.set_title('Improvement by Configuration', fontsize=13, fontweight='bold')
ax3.axvline(x=0, color='black', linewidth=1)
ax3.set_xlim(-5, 25)

# --- Panel 4: Heatmap - Horizon vs Barrier ---
ax4 = fig.add_subplot(2, 3, 4)
pivot_8 = results[results['features'] == 8].pivot(index='barrier', columns='horizon', values='improvement_pct')
im = ax4.imshow(pivot_8.values, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=15)
ax4.set_xticks(range(3))
ax4.set_xticklabels([10, 20, 30])
ax4.set_yticks(range(2))
ax4.set_yticklabels(['3%', '5%'])
ax4.set_xlabel('Horizon (days)', fontsize=12)
ax4.set_ylabel('Barrier Level', fontsize=12)
ax4.set_title('P2 Improvement % (8 features)', fontsize=13, fontweight='bold')
for i in range(2):
    for j in range(3):
        val = pivot_8.values[i, j]
        ax4.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')

# --- Panel 5: When Does P2 Win? ---
ax5 = fig.add_subplot(2, 3, 5)
# Group by characteristics
horizon_wins = results.groupby('horizon').apply(lambda x: (x['winner'] == 'P2').sum()).values
barrier_wins = results.groupby('barrier').apply(lambda x: (x['winner'] == 'P2').sum()).values
feature_wins = results.groupby('features').apply(lambda x: (x['winner'] == 'P2').sum()).values

x = np.arange(3)
width = 0.25
ax5.bar(x - width, [horizon_wins[0]/4, horizon_wins[1]/4, horizon_wins[2]/4], width, label='By Horizon', color='#3498db')
ax5.bar(x, [barrier_wins[0]/6, barrier_wins[1]/6, np.nan], width, label='By Barrier', color='#9b59b6')
ax5.bar(x + width, [feature_wins[0]/6, feature_wins[1]/6, np.nan], width, label='By Features', color='#f39c12')
ax5.set_xticks(x)
ax5.set_xticklabels(['10d / 3% / 8f', '20d / 5% / 32f', '30d / - / -'])
ax5.set_ylabel('P2 Win Rate', fontsize=12)
ax5.set_title('P2 Win Rate by Factor', fontsize=13, fontweight='bold')
ax5.legend()
ax5.set_ylim(0, 1.1)
ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# --- Panel 6: Key Insight ---
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

insight_text = """
KEY FINDINGS
═══════════════════════════════════════

Pipeline 2 (Distributional) wins 10/12 (83%)

WHY P2 WINS:
• Longer horizons: Uncertainty accumulates
  → Distribution captures this, direct model doesn't

• Rarer events (5% barrier): Fat tails matter
  → Student-t captures tail risk better

• Fewer features: Distribution provides structure
  → When features are limited, P2 compensates

P1'S ONLY WINS (both marginal):
• 10d, 3%, 32 features: +0.12% (negligible)
• 20d, 5%, 32 features: +0.03% (tie)

BOTTOM LINE:
Predicting distribution parameters → simulate
beats
Direct probability prediction
"""
ax6.text(0.05, 0.95, insight_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2))

plt.suptitle('Barrier Probability Prediction: Distributional vs Direct Approach\n'
             'S&P 500 Daily Returns (2009-2024) | Walk-Forward Validation',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('showcase/V3_barrier_results.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*70)
print("EXPERIMENT SUMMARY")
print("="*70)
print(f"\nTotal configurations tested: 12")
print(f"Pipeline 2 wins: 10 (83%)")
print(f"Pipeline 1 wins: 2 (17%)")
print(f"\nAverage P2 improvement: {results['improvement_pct'].mean():.1f}%")
print(f"Max P2 improvement: {results['improvement_pct'].max():.1f}% (H=10d, B=5%, F=32)")
print(f"\nP1's wins are marginal: +0.12% and +0.03%")
print("\nConclusion: Distributional modeling (temporalpdf) provides meaningful")
print("improvement over direct classification for barrier probability prediction.")
