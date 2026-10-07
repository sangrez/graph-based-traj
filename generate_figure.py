"""
Generates musit_model.png — architecture figure for the paper
"Transportation Mode Classification from GPS Trajectories Using
Graph Attention Networks"
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# ── Palette ──────────────────────────────────────────────────────────────────
C_INPUT   = '#D6EAF8'   # light blue  – raw inputs
C_H3      = '#D5F5E3'   # light green – H3 / graph construction
C_GAT     = '#EBF5FB'   # very light blue – GAT layers
C_GAT_BDR = '#2E86C1'   # blue border for GAT
C_WIDE    = '#FEF9E7'   # cream – wide branch
C_WIDE_BDR= '#D4AC0D'   # gold border
C_FUSE    = '#FDEDEC'   # light red – fusion
C_CLS     = '#F9F0FF'   # light purple – classifier
C_OUT     = '#EAFAF1'   # light green – output
ARROW     = '#555555'

def box(ax, x, y, w, h, label, sublabel=None,
        fc='#FFFFFF', ec='#333333', lw=1.5, fontsize=9,
        subfontsize=7.5, bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0),
            label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22,
                sublabel, ha='center', va='center',
                fontsize=subfontsize, color='#555555', zorder=4)

def arrow(ax, x1, y1, x2, y2, color=ARROW, lw=1.6, connectionstyle=None):
    style = connectionstyle if connectionstyle else "arc3,rad=0.0"
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle=style),
                zorder=5)

# ═══════════════════════════════════════════════════════════════════════════
# COLUMN POSITIONS
# ═══════════════════════════════════════════════════════════════════════════
# GPS input:  x=0.3
# H3 disc:    x=1.8
# Graph:      x=3.3
# GAT layers: x=5.0..7.6
# MaxPool:    x=8.5
# Wide net:   x=5.0 (bottom branch, y=1.5)
# Fusion:     x=10.1
# Classifier: x=11.8
# Output:     x=14.0

BH = 0.75   # standard box height
BW = 1.35   # standard box width

# ── TITLE ────────────────────────────────────────────────────────────────────
ax.text(8.0, 8.65, 'Proposed Wide-and-Deep GATv2 Architecture',
        ha='center', va='center', fontsize=13, fontweight='bold', color='#1A1A2E')

# ── SECTION LABELS ──────────────────────────────────────────────────────────
def section_label(ax, x, y, w, h, text, color):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='none', alpha=0.35, zorder=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h + 0.12, text, ha='center', va='bottom',
            fontsize=7.5, color='#333333', style='italic', zorder=2)

section_label(ax, 0.2,  3.6, 2.7, 3.8, 'Input Pipeline',     C_H3)
section_label(ax, 3.15, 3.6, 5.3, 3.8, 'Graph Attention Encoder', C_GAT)
section_label(ax, 4.85, 0.5, 3.3, 2.5, 'Wide Feature Network', C_WIDE)
section_label(ax, 8.85, 0.5, 4.7, 6.8, 'Classification Head',  C_CLS)

# ── INPUT COLUMN ─────────────────────────────────────────────────────────────
# GPS readings box
box(ax, 0.25, 6.6, BW+0.1, BH, 'GPS Trajectory',
    r'$(lat_i, lon_i, t_i)$', fc=C_INPUT, ec='#2874A6', bold=True)

# H3 discretisation
box(ax, 0.25, 5.4, BW+0.1, BH, 'H3 Discretization',
    'Resolution 9\n(~200 m cells)', fc=C_H3, ec='#1E8449', bold=False, fontsize=8.5, subfontsize=7)

# Sliding window
box(ax, 0.25, 4.15, BW+0.1, BH, 'Sliding Window',
    'W=100, S=50', fc=C_H3, ec='#1E8449', fontsize=8.5, subfontsize=7)

arrow(ax, 0.25+(BW+0.1)/2, 6.60,  0.25+(BW+0.1)/2, 6.15)  # GPS → H3
arrow(ax, 0.25+(BW+0.1)/2, 5.40,  0.25+(BW+0.1)/2, 5.10+0.15)  # H3 → window... wait

# Fix vertical arrows in input column
arrow(ax, 0.25+(BW+0.1)/2, 6.60,  0.25+(BW+0.1)/2, 6.16)
arrow(ax, 0.25+(BW+0.1)/2, 5.40,  0.25+(BW+0.1)/2, 4.92)
arrow(ax, 0.25+(BW+0.1)/2, 4.15,  0.25+(BW+0.1)/2, 3.80)

# ── GRAPH CONSTRUCTION ───────────────────────────────────────────────────────
GC_X = 1.9
box(ax, GC_X, 3.15, BW+0.25, BH*1.1,
    'Graph $G=(V,E)$',
    'Bidi + skip-1 edges\n6-dim edge attrs',
    fc=C_H3, ec='#1E8449', fontsize=8.5, subfontsize=6.8)

# Draw a tiny hexagonal graph icon
hx, hy = GC_X + 0.2, 3.22
r = 0.12
for angle, na, nb in [(0,1,2),(60,2,3),(120,3,4),(180,4,5),(240,5,0),(300,0,1)]:
    ax.annotate('', xy=(hx + r*np.cos(np.radians(nb*60+30)),
                        hy + r*np.sin(np.radians(nb*60+30))),
                xytext=(hx + r*np.cos(np.radians(na*60+30)),
                        hy + r*np.sin(np.radians(na*60+30))),
                arrowprops=dict(arrowstyle='-', color='#1E8449', lw=0.8), zorder=6)

# Arrow: input col → graph construction
arrow(ax, 0.25+(BW+0.1)/2, 3.80, GC_X, 3.60)

# Also: Wide features branch splits from graph
# Arrow graph → Wide feature entry (below)
arrow(ax, GC_X + (BW+0.25)/2, 3.15,
          GC_X + (BW+0.25)/2, 2.78,
      connectionstyle='arc3,rad=0.0')

# ── GAT LAYERS ───────────────────────────────────────────────────────────────
GAT_Y = 5.5
GAT_STARTS = [3.3, 5.05, 6.8]
GAT_HEADS  = ['4 heads', '2 heads', '1 head']
GAT_LABELS = ['GATv2Conv\nLayer 1', 'GATv2Conv\nLayer 2', 'GATv2Conv\nLayer 3']

prev_right = None
for i, (gx, heads, lbl) in enumerate(zip(GAT_STARTS, GAT_HEADS, GAT_LABELS)):
    box(ax, gx, GAT_Y, 1.55, 1.1, lbl, heads,
        fc=C_GAT, ec=C_GAT_BDR, lw=2.0, fontsize=8.5, subfontsize=7.5)
    if prev_right is not None:
        arrow(ax, prev_right, GAT_Y + 0.55, gx, GAT_Y + 0.55)
    prev_right = gx + 1.55

# Arrow: Graph → GAT layer 1
arrow(ax, GC_X + (BW+0.25), 3.65,   3.3, GAT_Y + 0.55,
      connectionstyle='arc3,rad=-0.25')

# Residual skip arrows (layer 1→3)
ax.annotate('', xy=(GAT_STARTS[1], GAT_Y + 1.1 + 0.05),
            xytext=(GAT_STARTS[0], GAT_Y + 1.1 + 0.05),
            arrowprops=dict(arrowstyle='->', color='#5D6D7E', lw=1.2,
                            connectionstyle='arc3,rad=-0.35'), zorder=5)
ax.annotate('', xy=(GAT_STARTS[2], GAT_Y + 1.1 + 0.05),
            xytext=(GAT_STARTS[1], GAT_Y + 1.1 + 0.05),
            arrowprops=dict(arrowstyle='->', color='#5D6D7E', lw=1.2,
                            connectionstyle='arc3,rad=-0.35'), zorder=5)
ax.text(5.0, 7.05, 'residual', fontsize=6.5, color='#5D6D7E', ha='center')

# ELU label on GAT arrows
ax.text(4.15, GAT_Y + 0.35, 'ELU', fontsize=6.5, color='#2E86C1', ha='center')
ax.text(5.90, GAT_Y + 0.35, 'ELU', fontsize=6.5, color='#2E86C1', ha='center')

# ── GLOBAL MAX POOL ──────────────────────────────────────────────────────────
POOL_X = 8.55
box(ax, POOL_X, GAT_Y, 1.4, 1.1, 'GlobalMaxPool',
    r'$h_G \in \mathbb{R}^{128}$',
    fc='#D7BDE2', ec='#7D3C98', lw=2.0, fontsize=8.5, subfontsize=7.5)
arrow(ax, GAT_STARTS[-1] + 1.55, GAT_Y + 0.55, POOL_X, GAT_Y + 0.55)

# ── WIDE FEATURE NETWORK ─────────────────────────────────────────────────────
WF_X = 3.4
WF_Y = 1.55

box(ax, WF_X - 1.0, WF_Y + 0.05, 1.3, BH*1.1,
    'Kinematic\nFeatures',
    r'$\mathbf{f} \in \mathbb{R}^{6}$',
    fc=C_WIDE, ec=C_WIDE_BDR, fontsize=8, subfontsize=7.5)

# Feature labels
feat_names = [r'$\hat{v}_{85}$', r'$r_{\rm stop}$', r'$\hat{\sigma}^2_v$',
              r'$\overline{\Delta\theta}$', r'$s_{\rm traj}$', r'$\hat{\bar{a}}$']
for fi, fn in enumerate(feat_names):
    ax.text(WF_X - 0.35, WF_Y + 0.12 + fi * 0.12, fn,
            fontsize=6, ha='center', va='bottom', color='#7D6608', zorder=4)

box(ax, WF_X + 0.5, WF_Y, 1.3, BH, 'FC (128)\nBN + ELU',
    fc=C_WIDE, ec=C_WIDE_BDR, fontsize=8)
box(ax, WF_X + 2.0, WF_Y, 1.3, BH, 'FC (128)\nBN + ELU',
    fc=C_WIDE, ec=C_WIDE_BDR, fontsize=8)

arrow(ax, WF_X - 0.0 + 0.3, WF_Y + BH/2, WF_X + 0.5, WF_Y + BH/2)
arrow(ax, WF_X + 0.5 + 1.3,  WF_Y + BH/2, WF_X + 2.0, WF_Y + BH/2)

# Label on wide output
ax.text(WF_X + 2.65 + 0.15, WF_Y + BH/2 + 0.22,
        r'$z \in \mathbb{R}^{128}$', fontsize=7, color='#7D6608', ha='center')

# Arrow: Graph → Wide features input
arrow(ax, GC_X + (BW+0.25)/2, 2.78,
          GC_X + (BW+0.25)/2, WF_Y + BH + 0.08,
      connectionstyle='arc3,rad=0.0')
arrow(ax, GC_X + (BW+0.25)/2, WF_Y + BH/2,
          WF_X - 0.0 + 0.0, WF_Y + BH/2)

# ── FUSION ───────────────────────────────────────────────────────────────────
FUSE_X = 10.2
FUSE_Y = 3.8

box(ax, FUSE_X, FUSE_Y, 1.3, BH*1.1,
    'Concat\nFusion',
    r'$h_f = [h_G \Vert z]$',
    fc=C_FUSE, ec='#C0392B', lw=2.0, fontsize=8.5, subfontsize=7.5)

# Arrow: MaxPool → Fusion
arrow(ax, POOL_X + 1.4, GAT_Y + 0.55,
          FUSE_X + 0.65, FUSE_Y + BH*1.1,
      connectionstyle='arc3,rad=0.1')

# Arrow: Wide net → Fusion
arrow(ax, WF_X + 2.0 + 1.3, WF_Y + BH/2,
          FUSE_X + 0.65, FUSE_Y,
      connectionstyle='arc3,rad=-0.15')

# Label on fused vector
ax.text(FUSE_X + 0.65, FUSE_Y - 0.28,
        r'$h_f \in \mathbb{R}^{256}$', fontsize=7, color='#C0392B', ha='center')

# ── CLASSIFICATION HEAD ───────────────────────────────────────────────────────
CLS_X = 11.85
cls_boxes = [
    (CLS_X, 5.6,  'FC (512)\nBN + ELU\nDO 0.4'),
    (CLS_X, 4.35, 'FC (256)\nBN + ELU\nDO 0.3'),
    (CLS_X, 3.10, 'FC (128)\nBN + ELU\nDO 0.2'),
    (CLS_X, 1.85, 'FC → 5\nSoftmax'),
]
prev_y = None
for (cx, cy, cl) in cls_boxes:
    box(ax, cx, cy, 1.3, 0.95, cl,
        fc=C_CLS, ec='#6C3483', lw=1.8, fontsize=7.5)
    if prev_y is not None:
        arrow(ax, cx + 0.65, prev_y, cx + 0.65, cy + 0.95)
    prev_y = cy

# Arrow: Fusion → Classifier
arrow(ax, FUSE_X + 1.3, FUSE_Y + BH*1.1/2,
          CLS_X,         5.6 + 0.475,
      connectionstyle='arc3,rad=0.0')

# ── OUTPUT ───────────────────────────────────────────────────────────────────
OUT_X = 14.15
OUT_Y = 1.85
classes = ['Walk', 'Bike', 'Bus', 'Car', 'Rail']
colors  = ['#76D7C4', '#7FB3D3', '#F0B27A', '#EC7063', '#A9CCE3']
for ci, (cls, col) in enumerate(zip(classes, colors)):
    by = OUT_Y + ci * 0.72
    rect = FancyBboxPatch((OUT_X, by), 1.4, 0.58,
                          boxstyle="round,pad=0.05",
                          facecolor=col, edgecolor='#333333',
                          linewidth=1.0, alpha=0.85, zorder=3)
    ax.add_patch(rect)
    ax.text(OUT_X + 0.70, by + 0.29, cls,
            ha='center', va='center', fontsize=8.5, fontweight='bold', zorder=4)

# Arrow: last FC → output
arrow(ax, CLS_X + 1.3, 1.85 + 0.29,
          OUT_X,        1.85 + 0.29)

# Arrow goes to midpoint of output column
ax.annotate('', xy=(OUT_X, OUT_Y + 2.5*0.72/2 + 0.29),
            xytext=(CLS_X + 1.3, 1.85 + 0.29 + 1.0),
            arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.5,
                            connectionstyle='arc3,rad=-0.2'), zorder=5)

ax.text(OUT_X + 0.70, OUT_Y + 5*0.72 + 0.15,
        'Mode\nPrediction', ha='center', va='bottom',
        fontsize=8, fontweight='bold', color='#1A1A2E')

# ── EDGE FEATURE LABEL ───────────────────────────────────────────────────────
ax.text(5.35, 5.2,
        r'edge attr $\mathbf{e}_{ij}$: [dist, $\Delta t$, speed, accel, $\Delta\theta$, speed-ratio]',
        ha='center', va='center', fontsize=6.8, color='#2E86C1',
        bbox=dict(boxstyle='round,pad=0.2', fc='#EBF5FB', ec='#2E86C1', lw=0.8, alpha=0.9),
        zorder=6)

# ── PHASE ANNOTATION ─────────────────────────────────────────────────────────
ax.annotate('', xy=(8.5, 0.28), xytext=(0.2, 0.28),
            arrowprops=dict(arrowstyle='-', color='#7F8C8D', lw=0.8,
                            linestyle='dashed'), zorder=2)
ax.text(4.35, 0.12, 'Phase 1: Pretrain on User-ID (cross-entropy, all segments)',
        ha='center', va='center', fontsize=7, color='#7F8C8D', style='italic')

ax.annotate('', xy=(16.0, 0.28), xytext=(8.5, 0.28),
            arrowprops=dict(arrowstyle='-', color='#7F8C8D', lw=0.8,
                            linestyle='dashed'), zorder=2)
ax.text(12.25, 0.12, 'Phase 2: Fine-tune on Mode Labels (Focal Loss, labeled only)',
        ha='center', va='center', fontsize=7, color='#7F8C8D', style='italic')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('/workspace/musit_model.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: musit_model.png")
