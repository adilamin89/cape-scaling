"""
Generate fig9_frontier.png for paper - 3-panel frontier figure
Panel (a): Cross-family scatter with regression, lab colors, phase shading + inset zoom
Panel (b): Anthropic within-family trajectory
Panel (c): h-field bar chart for all 20 models
"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

MODELS = [
    ("Claude 3.5 Son.",  49.0, 59.4, "Anthropic"),
    ("Claude 3.7 Son.",  62.3, 68.0, "Anthropic"),
    ("Haiku 4.5",        73.3, 71.0, "Anthropic"),
    ("Sonnet 4.5",       77.2, 83.4, "Anthropic"),
    ("Opus 4.5",         80.9, 87.0, "Anthropic"),
    ("Sonnet 4.6",       79.6, 74.1, "Anthropic"),
    ("Opus 4.6",         80.8, 91.3, "Anthropic"),
    ("Gem. 2.5 Pro",     63.8, 84.0, "Google"),
    ("Gem. 3 Flash",     78.0, 90.4, "Google"),
    ("Gem. 3 Pro",       76.2, 91.9, "Google"),
    ("Gem. 3.1 Pro",     80.6, 94.3, "Google"),
    ("GPT-4o",           33.2, 53.6, "OpenAI"),
    ("GPT-5",            74.9, 85.7, "OpenAI"),
    ("GPT-5.1",          76.3, 88.1, "OpenAI"),
    ("GPT-5.2 Pro",      80.0, 93.2, "OpenAI"),
    ("GPT-5.4",          77.2, 84.2, "OpenAI"),
    ("DeepSeek V3.2",    74.4, 79.9, "DeepSeek"),
    ("Kimi K2.5",        76.8, 87.6, "Moonshot"),
    ("Qwen 3.5-397B",   73.4, 88.4, "Alibaba"),
    ("MiniMax M2.5",     80.2, 85.0, "MiniMax"),
]

LAB_COLORS = {
    "Anthropic": "#E63946",
    "Google":    "#457B9D",
    "OpenAI":    "#2A9D8F",
    "DeepSeek":  "#E9C46A",
    "Moonshot":  "#F4A261",
    "Alibaba":   "#264653",
    "MiniMax":   "#6A0572",
}

LAB_MARKERS = {
    "Anthropic": "o",
    "Google":    "s",
    "OpenAI":    "D",
    "DeepSeek":  "^",
    "Moonshot":  "v",
    "Alibaba":   "p",
    "MiniMax":   "*",
}

swe = np.array([m[1] for m in MODELS])
gpqa = np.array([m[2] for m in MODELS])
labs = [m[3] for m in MODELS]
names = [m[0] for m in MODELS]

sl, ic, r, p, se = stats.linregress(swe, gpqa)
h_vals = gpqa - (sl * swe + ic)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={'width_ratios': [1.2, 0.8, 1.0]})
fig.patch.set_facecolor('white')

# ── Panel (a): Cross-family scatter ──
ax = axes[0]
x_fit = np.linspace(25, 88, 200)
y_fit = sl * x_fit + ic

# Phase shading
ax.fill_between(x_fit, y_fit, 100, alpha=0.06, color='green', label='Cooperative (h>0)')
ax.fill_between(x_fit, y_fit, 40, alpha=0.06, color='red', label='Tax excursion (h<0)')

# Regression line
ax.plot(x_fit, y_fit, 'k--', lw=1.5, alpha=0.7, label=f'GPQA = {sl:.2f}·SWE + {ic:.1f}')

# Plot points by lab with collision-avoiding labels
plotted_labs = set()
for i, (name, sv, gv, lab) in enumerate(MODELS):
    lbl = lab if lab not in plotted_labs else None
    plotted_labs.add(lab)
    ax.scatter(sv, gv, c=LAB_COLORS[lab], marker=LAB_MARKERS[lab],
               s=70, zorder=5, edgecolors='white', linewidths=0.5, label=lbl)

# Smart label placement with repulsion
label_positions = []
for i, (name, sv, gv, lab) in enumerate(MODELS):
    label_positions.append([sv, gv, name])

# Offset labels to avoid overlap
offsets = {}
offsets["GPT-4o"] = (-3, -6)
offsets["Claude 3.5 Son."] = (2, -5)
offsets["Claude 3.7 Son."] = (2, -5)
offsets["Gem. 2.5 Pro"] = (-12, 3)
offsets["Sonnet 4.6"] = (-14, -2)
offsets["Haiku 4.5"] = (-4, -6)
offsets["Opus 4.6"] = (-2, 4)
offsets["Gem. 3.1 Pro"] = (-2, 4)
offsets["GPT-5.2 Pro"] = (-4, 4)
offsets["Sonnet 4.5"] = (2, -5)
offsets["Opus 4.5"] = (-12, -3)
offsets["MiniMax M2.5"] = (2, -5)
offsets["Gem. 3 Pro"] = (-12, 4)
offsets["Gem. 3 Flash"] = (2, -4)
offsets["DeepSeek V3.2"] = (-12, -4)
offsets["Kimi K2.5"] = (2, 3)
offsets["Qwen 3.5-397B"] = (-12, -5)
offsets["GPT-5"] = (2, 2)
offsets["GPT-5.1"] = (-12, 2)
offsets["GPT-5.4"] = (2, -5)

for name, sv, gv, lab in MODELS:
    ox, oy = offsets.get(name, (2, 2))
    ax.annotate(name, (sv, gv), xytext=(ox, oy), textcoords='offset points',
                fontsize=5.5, color=LAB_COLORS[lab], alpha=0.9,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.3, alpha=0.4) if abs(ox) > 8 or abs(oy) > 4 else None)

ax.set_xlabel('SWE-bench Verified (%)', fontsize=10)
ax.set_ylabel('GPQA Diamond (%)', fontsize=10)
ax.set_title(f'(a) Cross-family: r = +{r:.2f}, p < 10⁻⁵', fontsize=10, fontweight='bold')
ax.set_xlim(28, 88)
ax.set_ylim(48, 100)
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.legend(fontsize=6, loc='lower right', framealpha=0.8)

# ── Inset zoom: crowded cluster region ──
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
axins = ax.inset_axes([0.02, 0.55, 0.45, 0.43])  # [x0, y0, width, height] in axes coords
zoom_swe = (70, 82)
zoom_gpqa = (70, 96)

# Regression line in inset
x_z = np.linspace(zoom_swe[0], zoom_swe[1], 100)
y_z = sl * x_z + ic
axins.plot(x_z, y_z, 'k--', lw=1, alpha=0.5)
axins.fill_between(x_z, y_z, 100, alpha=0.04, color='green')
axins.fill_between(x_z, y_z, 60, alpha=0.04, color='red')

# Plot cluster points in inset
for i, (name, sv, gv, lab) in enumerate(MODELS):
    if zoom_swe[0] <= sv <= zoom_swe[1] and zoom_gpqa[0] <= gv <= zoom_gpqa[1]:
        axins.scatter(sv, gv, c=LAB_COLORS[lab], marker=LAB_MARKERS[lab],
                      s=50, zorder=5, edgecolors='white', linewidths=0.4)
        # Label every point in inset
        ox, oy = offsets.get(name, (2, 2))
        # Tighter offsets for inset
        ox_in = 1.5 if ox > 0 else -1.5
        oy_in = 1.5 if oy > 0 else -1.5
        axins.annotate(name, (sv, gv), xytext=(ox_in*2, oy_in*2), textcoords='offset points',
                       fontsize=4.5, color=LAB_COLORS[lab], alpha=0.95,
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.2, alpha=0.3))

axins.set_xlim(*zoom_swe)
axins.set_ylim(*zoom_gpqa)
axins.set_xticks([72, 76, 80])
axins.set_yticks([75, 80, 85, 90, 95])
axins.tick_params(labelsize=5)
axins.grid(True, alpha=0.15, linewidth=0.3)
axins.set_title('cluster zoom', fontsize=5.5, pad=2, fontstyle='italic')
mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='gray', lw=0.6, alpha=0.4)

# ── Panel (b): Anthropic trajectory ──
ax = axes[1]
anth = [(m[0], m[1], m[2]) for m in MODELS if m[3] == "Anthropic"]
anth_swe = [m[1] for m in anth]
anth_gpqa = [m[2] for m in anth]
anth_names = [m[0] for m in anth]

ax.plot(x_fit, y_fit, 'k--', lw=1, alpha=0.4)
ax.fill_between(x_fit, y_fit, 100, alpha=0.04, color='green')
ax.fill_between(x_fit, y_fit, 40, alpha=0.04, color='red')

# Draw trajectory arrows
for i in range(len(anth) - 1):
    dx = anth_swe[i+1] - anth_swe[i]
    dy = anth_gpqa[i+1] - anth_gpqa[i]
    color = '#2A9D8F' if dy/max(abs(dx),0.1) > 0 else '#E63946'
    ax.annotate('', xy=(anth_swe[i+1], anth_gpqa[i+1]),
                xytext=(anth_swe[i], anth_gpqa[i]),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8, alpha=0.7))

ax.scatter(anth_swe, anth_gpqa, c='#E63946', s=60, zorder=5, edgecolors='white', linewidths=0.5)

# Label Anthropic models
anth_offsets = {
    "Claude 3.5 Son.": (3, -6),
    "Claude 3.7 Son.": (3, -6),
    "Haiku 4.5": (-4, -7),
    "Sonnet 4.5": (3, -6),
    "Opus 4.5": (-10, 4),
    "Sonnet 4.6": (-14, -3),
    "Opus 4.6": (-2, 5),
}
for name, sv, gv in anth:
    ox, oy = anth_offsets.get(name, (3, 3))
    ax.annotate(name.replace("Claude ", ""), (sv, gv), xytext=(ox, oy),
                textcoords='offset points', fontsize=6.5, color='#E63946',
                fontweight='bold')

# Highlight Sonnet 4.6 tax excursion
ax.annotate('tax\nexcursion', xy=(79.6, 74.1), xytext=(72, 65),
            fontsize=7, color='#E63946', fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color='#E63946', lw=1, alpha=0.6))

ax.set_xlabel('SWE-bench Verified (%)', fontsize=10)
ax.set_title('(b) Anthropic trajectory', fontsize=10, fontweight='bold')
ax.set_xlim(44, 86)
ax.set_ylim(55, 96)
ax.grid(True, alpha=0.2, linewidth=0.5)

# ── Panel (c): h-field bar chart ──
ax = axes[2]
sorted_idx = np.argsort(h_vals)
bar_colors = [LAB_COLORS[labs[i]] for i in sorted_idx]
bar_names = [names[i] for i in sorted_idx]
bar_h = h_vals[sorted_idx]

bars = ax.barh(range(len(bar_h)), bar_h, color=bar_colors, edgecolor='white', linewidth=0.3, height=0.7)
ax.axvline(0, color='black', lw=0.8, alpha=0.5)
ax.set_yticks(range(len(bar_names)))
ax.set_yticklabels(bar_names, fontsize=6)
ax.set_xlabel('h-field (pp)', fontsize=10)
ax.set_title('(c) Training-recipe h-field', fontsize=10, fontweight='bold')
ax.grid(True, axis='x', alpha=0.2, linewidth=0.5)

# Add value labels
for i, (h, name) in enumerate(zip(bar_h, bar_names)):
    side = 'left' if h > 0 else 'right'
    offset = 0.3 if h > 0 else -0.3
    ax.text(h + offset, i, f'{h:+.1f}', va='center', ha=side, fontsize=5, alpha=0.7)

plt.tight_layout()
import os
out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for dest in [os.path.join(out_dir, 'figures', 'fig9_frontier.png'),
             os.path.join(out_dir, 'fig9_frontier.png')]:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    plt.savefig(dest, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved {dest}")
print("Done — fig9_frontier.png with inset zoom")
plt.close()
