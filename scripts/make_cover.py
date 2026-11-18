# © 2026 Computational Classnotes · CC BY-NC-SA 4.0
# Cover art for Polymer Neural Nets: a glowing bead-spring chain, its graph edges and a neural-network motif.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from scipy.ndimage import gaussian_filter

W, H, DPI = 3840, 2160, 200
rng = np.random.default_rng(7)

# ---------- background: nebula ----------
yy, xx = np.mgrid[0:H // 4, 0:W // 4] / (H // 4)
xx = xx * (W / H)
bg = np.zeros((H // 4, W // 4, 3))
base = LinearSegmentedColormap.from_list("bg", ["#05010f", "#1a0533", "#081c3a", "#02131c"])
grad = base((0.55 * xx / xx.max() + 0.45 * yy))[..., :3]
bg += grad
blobs = [((0.35, 0.55), "#ff2bd6", 0.30), ((1.25, 0.35), "#00e5ff", 0.34), ((1.55, 0.80), "#7c3aed", 0.32),
         ((0.75, 0.15), "#ff7a00", 0.20), ((0.15, 0.95), "#00ffa3", 0.18), ((1.0, 0.65), "#ffd400", 0.12)]
for (cx, cy), col, s in blobs:
    c = np.array(matplotlib.colors.to_rgb(col))
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    noise = gaussian_filter(rng.random(g.shape), 6)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    bg += 0.75 * (g * (0.4 + 0.9 * noise))[..., None] * c
bg = np.clip(bg, 0, 1)

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.imshow(bg, extent=[0, W / H, 0, 1], interpolation="bicubic", zorder=0)
ax.set_xlim(0, W / H); ax.set_ylim(0, 1)

# stars
n = 1400
sx, sy = rng.random(n) * W / H, rng.random(n)
ax.scatter(sx, sy, s=rng.random(n) ** 3 * 6, c="white", alpha=0.7, linewidths=0, zorder=1)

def glow_line(x, y, color, lw=2, layers=8, alpha=0.08, z=3):
    for k in range(layers, 0, -1):
        ax.plot(x, y, color=color, lw=lw * (1 + 1.6 * k), alpha=alpha, solid_capstyle="round", zorder=z)
    ax.plot(x, y, color="white", lw=lw * 0.5, alpha=0.9, solid_capstyle="round", zorder=z + 0.1)

# ---------- the polymer: self-avoiding walk in 3D, projected ----------
def saw(N):
    pts = [np.zeros(3)]
    while len(pts) < N:
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        c = pts[-1] + v
        if len(pts) < 2 or np.min(np.linalg.norm(np.array(pts[:-1]) - c, axis=1)) > 1.0:
            pts.append(c)
    return np.array(pts)
P = saw(140)
P -= P.mean(0)
# smooth slightly for elegance
th = 0.6
R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
P = P @ R.T
scale = 0.36 / np.abs(P[:, :2]).max()
z = P[:, 2]; zn = (z - z.min()) / (z.max() - z.min())
persp = 1 / (1.35 - 0.5 * zn)
X = 1.30 + P[:, 0] * scale * persp
Y = 0.40 + P[:, 1] * scale * persp
hue = np.linspace(0, 0.92, len(P))
cols = hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.85), np.ones_like(hue)], 1))

# graph edges: non-bonded neighbours inside a cutoff
D = np.linalg.norm(P[:, None] - P[None], axis=2)
for i in range(len(P)):
    for j in range(i + 2, len(P)):
        if D[i, j] < 1.7:
            cc = (cols[i] + cols[j]) / 2
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color=cc, lw=4, alpha=0.12, zorder=2)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color=cc, lw=1.2, alpha=0.85, ls=(0, (3, 2)), zorder=2.1)
# bonds
for i in range(len(P) - 1):
    glow_line(X[i:i + 2], Y[i:i + 2], cols[i], lw=1.6 + 2.0 * zn[i], layers=6, alpha=0.07, z=3 + zn[i])
# beads with halos, back to front
for i in np.argsort(z):
    r = 9 + 26 * zn[i]
    for k, a in [(7, 0.05), (4.5, 0.08), (2.6, 0.16), (1.6, 0.3)]:
        ax.scatter(X[i], Y[i], s=(r * k) ** 2 / 9, color=cols[i], alpha=a, linewidths=0, zorder=4 + zn[i])
    ax.scatter(X[i], Y[i], s=r ** 2, color=cols[i], linewidths=0, zorder=4.5 + zn[i])
    ax.scatter(X[i] - 0.0012 * r * 0.25, Y[i] + 0.0015 * r * 0.25, s=(0.35 * r) ** 2, color="white",
               alpha=0.85, linewidths=0, zorder=4.6 + zn[i])

# ---------- neural-network motif, lower left ----------
layers = [5, 8, 8, 1]
nx0, ny0, dx, hgt = 0.15, 0.13, 0.19, 0.38
nodes = []
for li, m in enumerate(layers):
    ys = ny0 + hgt * (np.arange(m) + 0.5) / m if m > 1 else np.array([ny0 + hgt / 2])
    nodes.append(np.stack([np.full(m, nx0 + li * dx), ys], 1))
for li in range(len(layers) - 1):
    for a in nodes[li]:
        for b in nodes[li + 1]:
            c = hsv_to_rgb([0.52 + 0.12 * li, 0.8, 1])
            ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=3, alpha=0.06, zorder=2)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=0.8, alpha=0.5, zorder=2.05)
for li, L in enumerate(nodes):
    c = hsv_to_rgb([0.85 - 0.18 * li, 0.75, 1])
    for k, a in [(5, 0.06), (3, 0.12), (1.8, 0.25)]:
        ax.scatter(L[:, 0], L[:, 1], s=(14 * k) ** 2 / 4, color=c, alpha=a, linewidths=0, zorder=3)
    ax.scatter(L[:, 0], L[:, 1], s=110, color=c, edgecolors="white", linewidths=0.8, zorder=3.5)

# ---------- aurora ribbons across the sky ----------
t = np.linspace(0, W / H, 900)
for k in range(7):
    ph = rng.random() * 6.28
    yk = 0.30 + 0.06 * k + 0.05 * np.sin(2.1 * t + ph) + 0.025 * np.sin(5.3 * t + 2 * ph)
    hk = (0.78 + 0.09 * k) % 1
    for w, a in [(60, 0.025), (30, 0.04), (12, 0.06), (3, 0.12)]:
        ax.plot(t, yk, color=hsv_to_rgb([hk, 0.9, 1]), lw=w, alpha=a, solid_capstyle="round", zorder=1.5)

# ---------- message-passing ripples around a few beads ----------
for i in rng.choice(len(P), 7, replace=False):
    c = cols[i]
    for q, rad in enumerate([0.018, 0.032, 0.048, 0.066]):
        circ = plt.Circle((X[i], Y[i]), rad * (0.7 + 0.6 * zn[i]), fill=False, ec=c, lw=2.2 - 0.4 * q,
                          alpha=0.55 - 0.12 * q, zorder=3.2)
        ax.add_patch(circ)

# ---------- title ----------
def glow_text(x, y, s, size, color, weight="bold", z=10, family="DejaVu Sans", ha="left"):
    for k, a in [(10, 0.05), (6, 0.08), (3, 0.15)]:
        ax.text(x, y, s, fontsize=size, color=color, alpha=a, weight=weight, family=family, ha=ha, va="center",
                zorder=z, path_effects=[matplotlib.patheffects.withStroke(linewidth=k * 2, foreground=color)])
    ax.text(x, y, s, fontsize=size, color="white", weight=weight, family=family, ha=ha, va="center", zorder=z + 1)
import matplotlib.patheffects
glow_text(0.10, 0.86, "POLYMER", 92, "#ff2bd6")
glow_text(0.10, 0.755, "NEURAL NETS", 92, "#00e5ff")
ax.text(0.105, 0.69, "graph networks that learn the forces of a polymer chain", fontsize=26, color="#ffe9a8",
        family="DejaVu Sans", style="italic", va="center", zorder=11)
ax.text(0.105, 0.06, "Dara O Shayda  ·  Noor Hoshino", fontsize=20, color="#d6e4ff", family="DejaVu Sans",
        va="center", zorder=11)
ax.text(W / H - 0.03, 0.04, "© 2026 Computational Classnotes · CC BY-NC-SA 4.0", fontsize=15, color="#b9c6e8",
        ha="right", va="center", family="DejaVu Sans", zorder=11)
ax.text(W / H - 0.03, 0.95, "COMPUTATIONAL CLASSNOTES", fontsize=18, color="#ffd400", ha="right", va="center",
        family="DejaVu Sans", weight="bold", zorder=11)

fig.savefig("docs/cover.png", dpi=DPI)  # run from the repository folder
print("saved")
