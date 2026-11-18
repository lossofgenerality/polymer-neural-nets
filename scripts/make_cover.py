# © 2026 Computational Classnotes · CC BY-NC-SA 4.0
# Cover art for Polymer Neural Nets, an A4 portrait page (2480 x 3508 px, 300 dpi):
# a glowing bead-spring chain with its graph edges, and a neural-network motif, on a dark sky.
# Run from the repository folder: python3 scripts/make_cover.py  ->  docs/cover.png
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb, to_rgb
from scipy.ndimage import gaussian_filter

W, H, DPI = 2480, 3508, 300
A = W / H                      # the page spans x in [0, A], y in [0, 1]
rng = np.random.default_rng(7)

# ---------- background: a dark, low-contrast nebula, so the drawing stays readable ----------
h4, w4 = H // 4, W // 4
yy, xx = np.mgrid[0:h4, 0:w4] / h4
xx = xx * A
bg = LinearSegmentedColormap.from_list("bg", ["#04020c", "#0b0620", "#06122a", "#020a12"])(0.5 * xx / A + 0.5 * (1 - yy))[..., :3]
for (cx, cy), col, s in [((0.12, 0.88), "#ff2bd6", 0.20), ((0.62, 0.80), "#00e5ff", 0.22), ((0.55, 0.18), "#7c3aed", 0.26),
                         ((0.05, 0.30), "#00ffa3", 0.16), ((0.70, 0.50), "#ff7a00", 0.14)]:
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    noise = gaussian_filter(rng.random(g.shape), 8); noise = (noise - noise.min()) / np.ptp(noise)
    bg += 0.22 * (g * (0.3 + 0.8 * noise))[..., None] * np.array(to_rgb(col))
bg = np.clip(bg, 0, 1)

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.imshow(bg, extent=[0, A, 0, 1], interpolation="bicubic", zorder=0)
ax.set_xlim(0, A); ax.set_ylim(0, 1)
n = 900
ax.scatter(rng.random(n) * A, rng.random(n), s=rng.random(n) ** 3 * 4, c="white", alpha=0.5, linewidths=0, zorder=1)

def glow_line(x, y, color, lw=2.0, layers=6, alpha=0.07, z=3):
    for k in range(layers, 0, -1):
        ax.plot(x, y, color=color, lw=lw * (1 + 1.5 * k), alpha=alpha, solid_capstyle="round", zorder=z)
    ax.plot(x, y, color="white", lw=lw * 0.5, alpha=0.9, solid_capstyle="round", zorder=z + 0.01)

# ---------- the polymer: a self-avoiding walk in 3D, seen in perspective ----------
def saw(N):
    pts = [np.zeros(3)]
    while len(pts) < N:
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        c = pts[-1] + v
        if len(pts) < 2 or np.min(np.linalg.norm(np.array(pts[:-1]) - c, axis=1)) > 1.0:
            pts.append(c)
    return np.array(pts)
P = saw(120); P -= P.mean(0)
th = 0.6
P = P @ np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]).T
z = P[:, 2]; zn = (z - z.min()) / np.ptp(z)
persp = 1 / (1.35 - 0.5 * zn)
sc = 0.30 / np.abs(P[:, :2]).max()
X = 0.5 * A + P[:, 0] * sc * persp
Y = 0.49 + P[:, 1] * sc * persp
hue = np.linspace(0, 0.92, len(P))
cols = hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.85), np.ones_like(hue)], 1))

D = np.linalg.norm(P[:, None] - P[None], axis=2)          # graph edges: non-bonded pairs inside a cutoff
for i in range(len(P)):
    for j in range(i + 2, len(P)):
        if D[i, j] < 1.7:
            cc = (cols[i] + cols[j]) / 2
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color=cc, lw=5, alpha=0.10, zorder=2)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color=cc, lw=1.3, alpha=0.9, ls=(0, (3, 2)), zorder=2.1)
for i in range(len(P) - 1):                                  # bonds
    glow_line(X[i:i + 2], Y[i:i + 2], cols[i], lw=1.4 + 1.8 * zn[i], z=3 + zn[i])
for i in rng.choice(len(P), 6, replace=False):               # message-passing ripples
    for q, rad in enumerate([0.016, 0.028, 0.042, 0.058]):
        ax.add_patch(plt.Circle((X[i], Y[i]), rad * (0.7 + 0.6 * zn[i]), fill=False, ec=cols[i],
                                lw=2.0 - 0.4 * q, alpha=0.6 - 0.13 * q, zorder=3.2))
for i in np.argsort(z):                                      # beads, back to front
    r = 5 + 12 * zn[i]
    for k, a in [(6, 0.04), (4, 0.08), (2.4, 0.16), (1.5, 0.3)]:
        ax.scatter(X[i], Y[i], s=(r * k) ** 2 / 9, color=cols[i], alpha=a, linewidths=0, zorder=4 + zn[i])
    ax.scatter(X[i], Y[i], s=r ** 2, color=cols[i], linewidths=0, zorder=4.5 + zn[i])
    ax.scatter(X[i] - 0.0003 * r, Y[i] + 0.0003 * r, s=(0.35 * r) ** 2, color="white", alpha=0.85,
               linewidths=0, zorder=4.6 + zn[i])

# ---------- neural-network motif, below the chain ----------
layers = [5, 8, 8, 8, 1]
x0, x1, y0, hgt = 0.10, A - 0.10, 0.11, 0.15
nodes = []
for li, m in enumerate(layers):
    xl = x0 + (x1 - x0) * li / (len(layers) - 1)
    ys = y0 + hgt * (np.arange(m) + 0.5) / m if m > 1 else np.array([y0 + hgt / 2])
    nodes.append(np.stack([np.full(m, xl), ys], 1))
for li in range(len(layers) - 1):
    c = hsv_to_rgb([0.50 + 0.10 * li, 0.8, 1])
    for a in nodes[li]:
        for b in nodes[li + 1]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=3, alpha=0.05, zorder=2)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=0.8, alpha=0.55, zorder=2.05)
for li, L in enumerate(nodes):
    c = hsv_to_rgb([0.88 - 0.16 * li, 0.75, 1])
    for k, a in [(5, 0.06), (3, 0.12), (1.8, 0.25)]:
        ax.scatter(L[:, 0], L[:, 1], s=(12 * k) ** 2 / 4, color=c, alpha=a, linewidths=0, zorder=3)
    ax.scatter(L[:, 0], L[:, 1], s=90, color=c, edgecolors="white", linewidths=0.8, zorder=3.5)

# ---------- text ----------
def glow_text(x, y, s, size, color, z=10, ha="center"):
    for k, a in [(10, 0.05), (6, 0.08), (3, 0.15)]:
        ax.text(x, y, s, fontsize=size, color=color, alpha=a, weight="bold", family="DejaVu Sans", ha=ha,
                va="center", zorder=z, path_effects=[pe.withStroke(linewidth=k * 2, foreground=color)])
    ax.text(x, y, s, fontsize=size, color="white", weight="bold", family="DejaVu Sans", ha=ha, va="center", zorder=z + 1)
cx = A / 2
ax.text(cx, 0.955, "COMPUTATIONAL CLASSNOTES", fontsize=15, color="#ffd400", ha="center", va="center",
        family="DejaVu Sans", weight="bold", zorder=11)
glow_text(cx, 0.885, "POLYMER", 54, "#ff2bd6")
glow_text(cx, 0.825, "NEURAL NETS", 54, "#00e5ff")
ax.text(cx, 0.775, "graph networks that learn the forces of a polymer chain", fontsize=15, color="#ffe9a8",
        family="DejaVu Sans", style="italic", ha="center", va="center", zorder=11)
ax.text(cx, 0.058, "Dara O Shayda  ·  Noor Hoshino (星野 光)", fontsize=15, color="#d6e4ff",
        family=["DejaVu Sans", "Noto Sans CJK JP"], ha="center", va="center", zorder=11)
ax.text(cx, 0.030, "© 2026 Computational Classnotes · CC BY-NC-SA 4.0", fontsize=10, color="#9fb0d8",
        ha="center", va="center", family="DejaVu Sans", zorder=11)

fig.savefig("docs/cover.png", dpi=DPI)
