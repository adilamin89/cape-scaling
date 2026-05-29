#!/usr/bin/env python3
"""Generate cascade overview schematic for arXiv papers."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-1.2, 3.8)
ax.axis('off')

# Colors
tax_c = '#ef476f'
bonus_c = '#06d6a0'
trans_c = '#ffd166'
frontier_c = '#4cc9f0'
pred_c = '#7b2fff'

# Nc boxes
boxes = [
    (0, 'Nc1', '~3.5B', 'HS-TQA\nsign flip', 'Width, curation,\narchitecture', tax_c, bonus_c),
    (3.5, 'Nc2', '~30-72B', 'Cooperation\ncrashes 59%', 'Capacity\nlimit', bonus_c, trans_c),
    (7, 'Nc3', '~114B', 'SWE saturates,\nHLE activates', 'Benchmark\nrotation', trans_c, frontier_c),
    (10.5, 'Nc4', '~200-400B', 'IFEval saturates,\nnext axis TBD', 'Predicted', frontier_c, pred_c),
]

for x, label, scale, event, lever, c_before, c_after in boxes:
    # Before-zone
    rect_before = mpatches.FancyBboxPatch(
        (x - 0.3, 0.8), 1.2, 2.2,
        boxstyle="round,pad=0.1",
        facecolor=c_before, alpha=0.15, edgecolor=c_before, linewidth=1.5
    )
    ax.add_patch(rect_before)

    # After-zone
    rect_after = mpatches.FancyBboxPatch(
        (x + 1.1, 0.8), 1.2, 2.2,
        boxstyle="round,pad=0.1",
        facecolor=c_after, alpha=0.15, edgecolor=c_after, linewidth=1.5
    )
    ax.add_patch(rect_after)

    # Transition marker
    ax.annotate('', xy=(x + 1.1, 1.9), xytext=(x + 0.9, 1.9),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # Labels
    ax.text(x + 1.0, 3.3, label, ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#222')
    ax.text(x + 1.0, 3.05, scale, ha='center', va='bottom',
            fontsize=9, color='#666', style='italic')
    ax.text(x + 1.0, 1.9, event, ha='center', va='center',
            fontsize=8, color='#333', linespacing=1.3)
    ax.text(x + 1.0, 0.5, lever, ha='center', va='top',
            fontsize=7.5, color='#555', linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#ccc', alpha=0.8))

# Title
ax.text(6.75, 3.75, 'The Capability Cascade: Four Transitions, One Pattern',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color='#111')

# Bottom labels
ax.text(-0.3, -0.1, 'Phase:', fontsize=9, fontweight='bold', color='#444')
labels = [
    (0.5, 'Alignment\nTax', tax_c),
    (2.5, 'Alignment\nBonus', bonus_c),
    (5.5, 'Cooperative\n(Nc2 crash)', trans_c),
    (8.5, 'Frontier\nCooperative', frontier_c),
    (12, 'Next\nGeneration', pred_c),
]
for x, text, c in labels:
    ax.text(x, -0.2, text, ha='center', va='top', fontsize=7.5,
            color=c, fontweight='bold', linespacing=1.2)

# Bottom pattern description
ax.text(6.75, -1.0,
        'Each transition follows the same pattern: old benchmark axes lock together, '
        'coupling changes character, new axes emerge.',
        ha='center', va='top', fontsize=8.5, color='#666', style='italic')

plt.tight_layout()
plt.savefig('/Users/adilamin/Documents/research_2026/scaliwng_laws/cape-push-ready/fig_cascade_overview.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved fig_cascade_overview.png at 300 DPI")
plt.close()
