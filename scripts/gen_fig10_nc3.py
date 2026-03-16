"""
Generate fig10_nc3.png for paper — 3-panel Nc3 evidence figure
Panel (a): GPQA vs IFEval scatter with regression + √ boundary
Panel (b): Benchmark spread comparison (Nc1 vs Nc2 vs Nc3 pattern)
Panel (c): Symmetry cascade diagram (Ising → U(1) → O(3) → O(4))
"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# ── Data ──
NC3_MODELS = [
    ("Opus 4.6",     91.3, 94.0, "Anthropic",  "#E63946"),
    ("Kimi K2.5",    87.6, 94.0, "Moonshot",   "#F4A261"),
    ("Qwen 3.5-397B",88.4, 92.6, "Alibaba",    "#264653"),
    ("MiniMax M2.5", 85.0, 87.5, "MiniMax",     "#6A0572"),
]

gpqa = np.array([m[1] for m in NC3_MODELS])
ife  = np.array([m[2] for m in NC3_MODELS])
names = [m[0] for m in NC3_MODELS]
colors = [m[4] for m in NC3_MODELS]

sl, ic, r, p, se = stats.linregress(gpqa, ife)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1.0, 0.8, 1.2]})
fig.patch.set_facecolor('white')

# ── Panel (a): GPQA vs IFEval with regression + √ boundary ──
ax = axes[0]
x_fit = np.linspace(82, 96, 200)
y_reg = sl * x_fit + ic
y_sqrt = np.sqrt(0.97 * x_fit / 100) * 100  # √ isocline (rescaled to %)

# Phase shading
ax.fill_between(x_fit, y_sqrt, 100, alpha=0.06, color='green')
ax.fill_between(x_fit, y_sqrt, 82, alpha=0.06, color='red')

# √ boundary
ax.plot(x_fit, y_sqrt, 'r--', lw=1.5, alpha=0.5, label=r'$\sqrt{0.97 \cdot \mathrm{GPQA}}$ boundary')
# Linear regression
ax.plot(x_fit, y_reg, 'b--', lw=1.5, alpha=0.5, label=f'IFEval = {sl:.2f}·GPQA + {ic:.1f}')

# Plot points
for i, (name, gv, iv, lab, col) in enumerate(NC3_MODELS):
    ax.scatter(gv, iv, c=col, s=120, zorder=5, edgecolors='white', linewidths=1)
    # Labels
    offsets = {"Opus 4.6": (-15, 5), "Kimi K2.5": (5, -8),
               "Qwen 3.5-397B": (-20, -10), "MiniMax M2.5": (5, 5)}
    ox, oy = offsets.get(name, (5, 5))
    ax.annotate(name, (gv, iv), xytext=(ox, oy), textcoords='offset points',
                fontsize=8, color=col, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.4) if abs(ox) > 10 else None)

# Highlight MiniMax below boundary
ax.annotate('below √ boundary\n(new tax opening)', xy=(85.0, 87.5), xytext=(83, 84),
            fontsize=7, color='#E63946', fontstyle='italic', ha='center',
            arrowprops=dict(arrowstyle='->', color='#E63946', lw=1, alpha=0.6))

ax.set_xlabel('GPQA Diamond (%)', fontsize=10)
ax.set_ylabel('IFEval (%)', fontsize=10)
ax.set_title(f'(a) Nc₃ coupling: r = +{r:.2f} (n=4)', fontsize=10, fontweight='bold')
ax.set_xlim(83, 96)
ax.set_ylim(83, 98)
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.legend(fontsize=7, loc='lower right', framealpha=0.8)

# ── Panel (b): Benchmark spread comparison ──
ax = axes[1]
transitions = ['Nc₁\n(HS/TQA)', 'Nc₂\n(SWE/GPQA)', 'Nc₃\n(GPQA/IFEval)']
old_spread = [4.9, 0.9, 9.3]  # Saturating benchmark spread
new_spread = [None, 6.5, 6.5]  # Activating benchmark spread (IFEval at Nc3, SWE at Nc2... hmm)

# Actually let me rethink this. The pattern is:
# At Nc1: HS/TQA saturate (spread ~4.9pp), SWE/GPQA activate
# At Nc2: SWE/GPQA... no, SWE saturates (0.9pp), IFEval activates (6.5pp)
# Better to show: "saturating axis" and "activating axis" at each Nc

sat_labels = ['HS/TQA\nsaturating', 'SWE-bench\nsaturating', 'IFEval\n(next to saturate?)']
sat_values = [4.9, 0.9, 6.5]  # spread of the saturating/active benchmark
sat_colors = ['#ffd166', '#06d6a0', '#4cc9f0']

bars = ax.bar(range(3), sat_values, color=sat_colors, edgecolor='white', linewidth=0.5, width=0.6)
ax.set_xticks(range(3))
ax.set_xticklabels(sat_labels, fontsize=8)
ax.set_ylabel('Benchmark spread (pp)', fontsize=10)
ax.set_title('(b) Saturation pattern at each Nc', fontsize=10, fontweight='bold')
ax.grid(True, axis='y', alpha=0.2, linewidth=0.5)

# Add labels on bars
for i, (v, c) in enumerate(zip(sat_values, sat_colors)):
    ax.text(i, v + 0.2, f'{v} pp', ha='center', va='bottom', fontsize=9, fontweight='bold', color=c)

# Add threshold line
ax.axhline(y=2, color='gray', linestyle='--', linewidth=1, alpha=0.4)
ax.text(2.4, 2.2, 'saturation\nthreshold', fontsize=7, color='gray', ha='right')

ax.set_ylim(0, 10)

# ── Panel (c): Symmetry cascade ──
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('(c) Symmetry cascade: group-subgroup chain', fontsize=10, fontweight='bold')

# Draw boxes for each Nc
boxes = [
    (0.3, 3.5, 'Nc₁ ≈ 3.5B', 'Ising (Z₂)', 'd_eff → 1', '1 axis', '#ffd166', 'Domain\nwalls'),
    (2.7, 3.5, 'Nc₂ ≈ 70–88B', 'U(1) / XY', 'd_eff → 2', '2 axes', '#4cc9f0', 'Vortex\nlines'),
    (5.1, 3.5, 'Nc₃ ≈ 114B', 'O(3) / Heis.', 'd_eff → 3', '3 axes', '#7b2fff', 'Hedgehog\ndefects'),
    (7.5, 3.5, 'Nc₄ ~ 200–400B', 'O(4)', 'd_eff → 4', '4 axes', '#8aa8c8', 'Topological\nprotection'),
]

for x, y, title, sym, deff, axes_n, col, defect in boxes:
    # Box
    rect = FancyBboxPatch((x, y-1.8), 2.0, 3.2, boxstyle="round,pad=0.1",
                           facecolor=col, alpha=0.08, edgecolor=col, linewidth=1.5)
    ax.add_patch(rect)
    # Title
    ax.text(x+1.0, y+1.2, title, ha='center', va='center', fontsize=8, fontweight='bold', color=col)
    # Symmetry
    ax.text(x+1.0, y+0.6, sym, ha='center', va='center', fontsize=9, fontweight='bold', color=col)
    # d_eff
    ax.text(x+1.0, y+0.0, deff, ha='center', va='center', fontsize=7, color='#555')
    # axes
    ax.text(x+1.0, y-0.5, axes_n, ha='center', va='center', fontsize=7, color='#555')
    # defects
    ax.text(x+1.0, y-1.2, defect, ha='center', va='center', fontsize=6.5, color='#888', fontstyle='italic')

# Arrows between boxes
for i in range(3):
    x_start = boxes[i][0] + 2.05
    x_end = boxes[i+1][0] - 0.05
    ax.annotate('', xy=(x_end, 3.5), xytext=(x_start, 3.5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

# Bottom annotation
ax.text(5.0, 0.3, 'Each transition expands the symmetry group.\n'
        'At O(3), topological protection of cooperative alignment becomes possible.',
        ha='center', va='center', fontsize=7.5, color='#666', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ddd', alpha=0.5))

plt.tight_layout()
plt.savefig('/sessions/nice-busy-lamport/cape-staging/fig10_nc3.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved fig10_nc3.png")
plt.close()
