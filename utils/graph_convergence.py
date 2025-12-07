"""
Génération de la Figure de Convergence pour l'analyse POPE-Adversarial
Figure combinée : (a) Uncertainty Decay + (b) Accuracy Improvement
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configuration de style pour publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


# ============================================================================
# DONNÉES EXPÉRIMENTALES
# ============================================================================

# Iterations (0 = baseline, 1-3 = refinement iterations)
iterations = np.array([0, 1, 2, 3])

# --- (a) UNCERTAINTY DECAY ---
# Mean uncertainty scores u_t
uncertainty_mean = np.array([0.52, 0.38, 0.31, 0.27])
# Standard deviation (CORRIGÉ: 0.15 à l'iteration 1)
uncertainty_std = np.array([0.18, 0.15, 0.13, 0.12])  # ← CORRECTION ICI

# --- (b) ACCURACY IMPROVEMENT ---
# Test accuracy (%) - From Table 1 (POPE-Adversarial)
accuracy = np.array([75.9, 78.7, 79.8, 80.6])

# Cumulative convergence rates (% of samples with u_t < 0.3)
convergence_rate = np.array([0, 35, 68, 82])


# ============================================================================
# CRÉATION DE LA FIGURE COMBINÉE
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle('Convergence Analysis on POPE-Adversarial Split', 
             fontsize=14, fontweight='bold', y=1.02)

# ============================================================================
# SUBPLOT (a): UNCERTAINTY DECAY
# ============================================================================

# Plot mean uncertainty with error bars
ax1.plot(iterations, uncertainty_mean, 
         marker='o', markersize=8, linewidth=2.5, 
         color='#E74C3C', label='Mean Uncertainty $u_t$',
         zorder=3)

# Add error bars (±1 std)
ax1.fill_between(iterations, 
                  uncertainty_mean - uncertainty_std,
                  uncertainty_mean + uncertainty_std,
                  alpha=0.25, color='#E74C3C', zorder=2)

# Add threshold line τ_u = 0.3
ax1.axhline(y=0.3, color='#7F8C8D', linestyle='--', 
            linewidth=2, label='Threshold $\\tau_u = 0.3$',
            zorder=1)

# Annotations for specific points
for i, (x, y) in enumerate(zip(iterations, uncertainty_mean)):
    ax1.annotate(f'{y:.2f}', 
                xy=(x, y), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center', fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor='#E74C3C',
                         alpha=0.8))

# Add delta annotations between iterations
deltas = np.diff(uncertainty_mean)
for i, delta in enumerate(deltas):
    mid_x = iterations[i] + 0.5
    mid_y = (uncertainty_mean[i] + uncertainty_mean[i+1]) / 2
    ax1.annotate(f'Δ={delta:.2f}', 
                xy=(mid_x, mid_y),
                fontsize=8, style='italic',
                color='#C0392B',
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='#FADBD8', 
                         alpha=0.7))

# Styling
ax1.set_xlabel('Iteration', fontweight='bold')
ax1.set_ylabel('Uncertainty Score $u_t$', fontweight='bold')
ax1.set_title('(a) Uncertainty Decay', fontweight='bold', loc='left')
ax1.set_xticks(iterations)
ax1.set_xticklabels(['0\n(Baseline)', '1', '2', '3'])
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax1.legend(loc='upper right', framealpha=0.95)
ax1.set_ylim([0.2, 0.6])

# Add background shading for diminishing returns region
ax1.axvspan(2, 3, alpha=0.1, color='gray', zorder=0)
ax1.text(2.5, 0.57, 'Diminishing\nReturns', 
         ha='center', va='top', fontsize=9, style='italic',
         color='gray')

# ============================================================================
# SUBPLOT (b): ACCURACY IMPROVEMENT WITH CONVERGENCE RATES
# ============================================================================

# Primary axis: Accuracy
color_accuracy = '#2ECC71'
ax2.plot(iterations, accuracy, 
         marker='s', markersize=8, linewidth=2.5, 
         color=color_accuracy, label='Test Accuracy',
         zorder=3)

# Add accuracy values as annotations
for i, (x, y) in enumerate(zip(iterations, accuracy)):
    ax2.annotate(f'{y:.1f}%', 
                xy=(x, y), 
                xytext=(0, -15),
                textcoords='offset points',
                ha='center', fontsize=9,
                fontweight='bold',
                color=color_accuracy,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor=color_accuracy,
                         alpha=0.8))

# Add gain annotations
gains = np.diff(accuracy)
gain_percentages = [60, 23, 17]  # % of total gain
for i, (gain, pct) in enumerate(zip(gains, gain_percentages)):
    mid_x = iterations[i] + 0.5
    mid_y = (accuracy[i] + accuracy[i+1]) / 2 + 0.5
    ax2.annotate(f'+{gain:.1f}pp\n({pct}%)', 
                xy=(mid_x, mid_y),
                fontsize=8,
                ha='center',
                color='#27AE60',
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='#D5F4E6', 
                         alpha=0.8))

# Secondary axis: Convergence Rate
ax2_right = ax2.twinx()
color_convergence = '#3498DB'
ax2_right.plot(iterations, convergence_rate, 
               marker='^', markersize=7, linewidth=2, 
               linestyle='--', color=color_convergence, 
               label='Convergence Rate',
               alpha=0.8, zorder=2)

# Add convergence rate annotations
for i, (x, y) in enumerate(zip(iterations[1:], convergence_rate[1:]), start=1):
    ax2_right.annotate(f'{y}%', 
                      xy=(x, y), 
                      xytext=(8, 0),
                      textcoords='offset points',
                      ha='left', fontsize=8,
                      color=color_convergence,
                      style='italic')

# Styling for primary axis (Accuracy)
ax2.set_xlabel('Iteration', fontweight='bold')
ax2.set_ylabel('Test Accuracy (%)', fontweight='bold', color=color_accuracy)
ax2.tick_params(axis='y', labelcolor=color_accuracy)
ax2.set_title('(b) Accuracy Improvement', fontweight='bold', loc='left')
ax2.set_xticks(iterations)
ax2.set_xticklabels(['0\n(Baseline)', '1', '2', '3'])
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')
ax2.set_ylim([74, 82])

# Styling for secondary axis (Convergence Rate)
ax2_right.set_ylabel('Cumulative Convergence Rate (%)', 
                     fontweight='bold', color=color_convergence)
ax2_right.tick_params(axis='y', labelcolor=color_convergence)
ax2_right.set_ylim([0, 100])

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper left', framealpha=0.95)

# Highlight first iteration (major gain)
rect = Rectangle((0.85, 75.5), 0.3, 3.5, 
                 linewidth=2, edgecolor='#E67E22', 
                 facecolor='none', linestyle=':', 
                 zorder=1)
ax2.add_patch(rect)
ax2.text(1.0, 74.5, 'Major\nGain', 
        ha='center', fontsize=8, style='italic',
        color='#E67E22', fontweight='bold')

# ============================================================================
# AJUSTEMENTS FINAUX
# ============================================================================

plt.tight_layout()

# Sauvegarder en haute résolution
plt.savefig('figures/convergence_combined.pdf', 
            dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('figures/convergence_combined.png', 
            dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("✅ Figure sauvegardée avec succès!")
print("   - PDF: figures/convergence_combined.pdf")
print("   - PNG: figures/convergence_combined.png")

# Afficher la figure
plt.show()

# ============================================================================
# STATISTIQUES RÉCAPITULATIVES (pour vérification)
# ============================================================================

print("\n📊 STATISTIQUES DE CONVERGENCE:")
print("="*50)

print("\n(a) Uncertainty Decay:")
for i in range(len(iterations)):
    print(f"  Iteration {iterations[i]}: u_t = {uncertainty_mean[i]:.2f} ± {uncertainty_std[i]:.2f}")
    if i > 0:
        delta = uncertainty_mean[i-1] - uncertainty_mean[i]
        print(f"    → Δu = {delta:.2f}")

print(f"\n  Total reduction: Δu = {uncertainty_mean[0] - uncertainty_mean[-1]:.2f}")
print(f"  Percentage decrease: {((uncertainty_mean[0] - uncertainty_mean[-1])/uncertainty_mean[0])*100:.1f}%")

print("\n(b) Accuracy Improvement:")
for i in range(len(iterations)):
    print(f"  Iteration {iterations[i]}: Accuracy = {accuracy[i]:.1f}%")
    if i > 0:
        gain = accuracy[i] - accuracy[i-1]
        pct_of_total = (gain / (accuracy[-1] - accuracy[0])) * 100
        print(f"    → Gain = +{gain:.1f}pp ({pct_of_total:.0f}% of total)")
        print(f"    → Convergence Rate = {convergence_rate[i]}%")

print(f"\n  Total improvement: {accuracy[-1] - accuracy[0]:.1f} percentage points")
print(f"  Final accuracy: {accuracy[-1]:.1f}%")

print("\n" + "="*50)
print("✨ Analyse terminée!")