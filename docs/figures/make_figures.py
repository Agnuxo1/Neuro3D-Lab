"""Generate the animated GIFs of the README from the repository's own code.

Each figure is computed with optlab or taken from the versioned results in
experiments/results/.
Usage: python docs/figures/make_figures.py   ->   docs/img/*.gif
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from optlab.propagation import angular_spectrum  # noqa: E402
from optlab.fresnel import oblique  # noqa: E402
from optlab.interferometer import mzi  # noqa: E402
from optlab.d2nn import DiffractiveNetwork  # noqa: E402
from optlab.stats import wilson_interval  # noqa: E402
import run_d2nn_digits as R  # noqa: E402  (geometry and encoding identical to the contract)

OUT = ROOT / "docs" / "img"
RESULTS = ROOT / "experiments" / "results"
LAM = 633e-9

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
CYAN = "#22d3ee"
VIOLET = "#a78bfa"
AMBER = "#fbbf24"
GREEN = "#34d399"
RED = "#f87171"
PANEL = "#161b22"
BORDER = "#30363d"

# Verdict labels as recorded in the result files (Spanish or English) -> English text shown in figures.
VERDICT_EN = {
    "cumple": "pass", "pass": "pass",
    "no cumple": "fail", "fail": "fail",
    "exploratorio": "exploratory", "exploratory": "exploratory",
    "pendiente": "pending", "pending": "pending",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BORDER, "font.family": "DejaVu Sans", "font.size": 10,
})


def make_anim(fig, update, frames, fps, name, dpi=72):
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    path = OUT / name
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"{name}: {path.stat().st_size // 1024} KB, {frames} frames")


def beam_width(I, x):
    Ix = I.sum(axis=0)
    return 2 * math.sqrt(np.sum(Ix * x**2) / np.sum(Ix))


def verdict_label(raw):
    """English label for a recorded verdict; a missing verdict is shown as pending."""
    if raw is None:
        return "pending"
    return VERDICT_EN.get(raw, raw)


# ---------------------------------------------------------------- header
def fig_header():
    N, dx, w0 = 512, 2e-6, 20e-6
    x = (np.arange(N) - (N - 1) / 2) * dx
    X, Y = np.meshgrid(x, x)
    u0 = np.exp(-(X**2 + Y**2) / w0**2).astype(complex)
    zR = math.pi * w0**2 / LAM
    zs = np.linspace(0.0, 1.5 * zR, 48)
    fields = [np.abs(angular_spectrum(u0, dx, LAM, z)) ** 2 for z in zs]
    w_num = [beam_width(F, x) for F in fields]
    zz = np.linspace(0, 1.5 * zR, 200)
    w_an = w0 * np.sqrt(1 + (zz / zR) ** 2)
    crop = slice(N // 2 - 70, N // 2 + 70)

    fig = plt.figure(figsize=(12, 4.6))
    fig.text(0.5, 0.93, "Neuro3D-Lab", ha="center", fontsize=28, weight="bold", color=FG)
    fig.text(0.5, 0.84, "Simulation-only optics lab in Blender for optical neural networks, "
             "with wave-optics calculations",
             ha="center", fontsize=12, color=CYAN)
    ax1 = fig.add_axes([0.03, 0.06, 0.40, 0.66])
    ax1.set_xticks([]); ax1.set_yticks([])
    im = ax1.imshow(fields[0][crop, crop] / fields[0].max(), cmap="magma", vmin=0, vmax=1)
    ax1.set_title("Propagated intensity  |U(x, y, z)|²", fontsize=10, color=MUTED)
    ax2 = fig.add_axes([0.52, 0.14, 0.44, 0.56])
    ax2.plot(zz * 1e3, w_an * 1e6, color=MUTED, lw=1.5, ls="--", label="analytic  w(z)")
    pts, = ax2.plot([], [], "o", color=CYAN, ms=5, label="numerical (optlab)")
    ax2.set_xlim(0, 1.5 * zR * 1e3); ax2.set_ylim(0, w_an.max() * 1.1 * 1e6)
    ax2.set_xlabel("z  [mm]"); ax2.set_ylabel("beam radius w  [µm]")
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    def update(i):
        im.set_data(fields[i][crop, crop] / fields[0].max())
        pts.set_data(zs[: i + 1] * 1e3, np.array(w_num[: i + 1]) * 1e6)
        return im, pts

    make_anim(fig, update, len(zs), 12, "header.gif", dpi=72)


# -------------------------------------------------------------- pipeline
def fig_pipeline():
    fig = plt.figure(figsize=(12, 4.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(0.5, 0.95, "Virtual bench data flow", ha="center", fontsize=16, weight="bold")

    def box(x0, title, lines, color):
        ax.add_patch(FancyBboxPatch((x0, 0.22), 0.27, 0.5, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc=PANEL, ec=color, lw=2))
        ax.text(x0 + 0.135, 0.66, title, ha="center", va="center", fontsize=12, weight="bold", color=color)
        for k, t in enumerate(lines):
            ax.text(x0 + 0.02, 0.55 - 0.065 * k, t, ha="left", va="center", fontsize=9, color=FG,
                    family="monospace")

    box(0.03, "Blender · bench", ["SOURCE  z = 0 m", "LAYER_1  z = 20 mm", "LAYER_2  z = 40 mm",
                                 "DETECTOR z = 60 mm", "heights → .npy"], CYAN)
    box(0.37, "optlab · physics", ["angular spectrum", "phase masks", "intensity |f|²",
                                  "regions → scores", "optlab.bench/1"], VIOLET)
    box(0.71, "Blender · result", ["float image", "emission at DETECTOR", "saved scale",
                                  "ray tracing", "check"], GREEN)
    arrows = []
    for x0, x1, label in ((0.30, 0.37, "bench.json\n+ heights"), (0.64, 0.71, "detector_\nintensity.npy")):
        a = FancyArrowPatch((x0, 0.47), (x1, 0.47), arrowstyle="-|>", mutation_scale=18, color=MUTED, lw=1.5)
        ax.add_patch(a)
        ax.text((x0 + x1) / 2, 0.40, label, ha="center", va="center", fontsize=7, color=MUTED)
    packets = [Circle((0.30, 0.47), 0.012, color=AMBER), Circle((0.64, 0.47), 0.012, color=AMBER)]
    for p in packets:
        ax.add_patch(p)
    ax.text(0.5, 0.11, "build_demo_bench  →  export_bench  →  simulate_bench  →  import_intensity",
            ha="center", fontsize=9, color=MUTED, family="monospace")

    frames = 60

    def update(i):
        if i < 30:
            t = i / 29
            packets[0].center = (0.30 + 0.07 * t, 0.47)
            packets[1].center = (0.64, 0.47)
        else:
            t = (i - 30) / 29
            packets[0].center = (0.37 - 0.0, 0.47) if False else (0.30 + 0.07, 0.47)
            packets[1].center = (0.64 + 0.07 * t, 0.47)
        return packets

    make_anim(fig, update, frames, 14, "pipeline.gif", dpi=72)


# ------------------------------------------------------------ propagation
def fig_propagation():
    N, dx, w0 = 512, 2e-6, 20e-6
    x = (np.arange(N) - (N - 1) / 2) * dx
    X, Y = np.meshgrid(x, x)
    u0 = np.exp(-(X**2 + Y**2) / w0**2).astype(complex)
    zR = math.pi * w0**2 / LAM
    zs = np.linspace(0.02 * zR, 1.5 * zR, 40)
    fields = [np.abs(angular_spectrum(u0, dx, LAM, z)) ** 2 for z in zs]
    w_num = np.array([beam_width(F, x) for F in fields])
    zz = np.linspace(0, 1.5 * zR, 200)
    w_an = w0 * np.sqrt(1 + (zz / zR) ** 2)
    crop = slice(N // 2 - 70, N // 2 + 70)

    fig = plt.figure(figsize=(12, 4.4))
    fig.text(0.5, 0.94, "Propagation of a Gaussian beam by the angular-spectrum method",
             ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.86, r"$U = \mathcal{F}^{-1}\{\,H\cdot\mathcal{F}\{u_0\}\}$,   "
             r"$H = \exp\!\left(i\,2\pi d\sqrt{1/\lambda^2 - f_x^2 - f_y^2}\right)$",
             ha="center", fontsize=12, color=CYAN)
    ax1 = fig.add_axes([0.03, 0.08, 0.36, 0.68]); ax1.set_xticks([]); ax1.set_yticks([])
    im = ax1.imshow(fields[0][crop, crop] / fields[0].max(), cmap="inferno", vmin=0, vmax=1)
    ax2 = fig.add_axes([0.50, 0.16, 0.46, 0.58])
    ax2.plot(zz * 1e3, w_an * 1e6, color=MUTED, ls="--", lw=1.5, label="exact Gaussian")
    pts, = ax2.plot([], [], "o", color=CYAN, ms=4, label="numerical (validation E1)")
    ax2.set_xlim(0, 1.5 * zR * 1e3); ax2.set_ylim(0, w_an.max() * 1.15 * 1e6)
    ax2.set_xlabel("z [mm]"); ax2.set_ylabel("w(z) [µm]")
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    txt = ax2.text(0.98, 0.05, "", transform=ax2.transAxes, ha="right", color=AMBER, fontsize=10)

    def update(i):
        im.set_data(fields[i][crop, crop] / fields[0].max())
        pts.set_data(zs[: i + 1] * 1e3, w_num[: i + 1] * 1e6)
        err = abs(w_num[i] / (w0 * math.sqrt(1 + (zs[i] / zR) ** 2)) - 1)
        txt.set_text(f"z = {zs[i] * 1e3:.2f} mm   relative error of w = {err:.1e}")
        return im, pts, txt

    make_anim(fig, update, len(zs), 12, "propagation.gif", dpi=72)


# ---------------------------------------------------------------- fresnel
def fig_fresnel():
    th = np.linspace(0, math.pi / 2 - 1e-3, 300)
    Rs = np.array([oblique(1.0, 1.5, t, "s")["R"] for t in th])
    Rp = np.array([oblique(1.0, 1.5, t, "p")["R"] for t in th])
    brew = math.degrees(math.atan(1.5))
    fig = plt.figure(figsize=(12, 4.4))
    fig.text(0.5, 0.94, "Fresnel reflection at the air → glass interface (n = 1.5)",
             ha="center", fontsize=15, weight="bold")
    ax = fig.add_axes([0.08, 0.15, 0.6, 0.7])
    ax.set_xlim(0, 90); ax.set_ylim(0, 1.02)
    ax.set_xlabel("angle of incidence θᵢ [°]"); ax.set_ylabel("reflectance R")
    ax.axvline(brew, color=AMBER, ls=":", lw=1.2)
    ax.text(brew + 1, 0.12, f"Brewster {brew:.1f}°\nR_p = 0", color=AMBER, fontsize=10)
    ls, = ax.plot([], [], color=CYAN, lw=2.2, label="R_s (perpendicular)")
    lp, = ax.plot([], [], color=VIOLET, lw=2.2, label="R_p (parallel)")
    dot, = ax.plot([], [], "o", color=FG, ms=7)
    ax.legend(frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    info = fig.add_axes([0.72, 0.15, 0.25, 0.7]); info.axis("off")
    info.text(0, 0.9, "Energy balance", fontsize=12, weight="bold", color=FG)
    info.text(0, 0.78, "R + T = 1", fontsize=12, color=GREEN, family="monospace")
    info.text(0, 0.62, "normal incidence (θ = 0):", fontsize=10, color=MUTED)
    n_norm = oblique(1.0, 1.5, 0.0, "s")["R"]
    info.text(0, 0.52, f"R = ((n₁−n₂)/(n₁+n₂))² = {n_norm:.4f}", fontsize=9, color=FG, family="monospace")
    info.text(0, 0.36, "Snell's law:", fontsize=10, color=MUTED)
    info.text(0, 0.27, "n₁ sin θᵢ = n₂ sin θₜ", fontsize=9, color=FG, family="monospace")
    info.text(0, 0.12, "Above 41.8° glass→air:\ntotal internal reflection (R = 1)", fontsize=9, color=MUTED)
    step = 2

    def update(i):
        k = min(len(th), (i + 1) * step)
        ls.set_data(np.degrees(th[:k]), Rs[:k]); lp.set_data(np.degrees(th[:k]), Rp[:k])
        dot.set_data([np.degrees(th[k - 1])], [Rp[k - 1]])
        return ls, lp, dot

    frames = len(th) // step
    make_anim(fig, update, frames, 14, "fresnel.gif", dpi=72)


# -------------------------------------------------------------------- mzi
def fig_mzi():
    th = np.linspace(0, 2 * math.pi, 120)
    P1 = np.array([abs(mzi(t, 0.0)[0, 0]) ** 2 for t in th])
    P2 = np.array([abs(mzi(t, 0.0)[1, 0]) ** 2 for t in th])
    fig = plt.figure(figsize=(12, 4.4))
    fig.text(0.5, 0.94, "Mach-Zehnder interferometer: power split versus phase θ",
             ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.86, r"$U(\theta,\phi)=B\,P(\theta)\,B\,P(\phi)$,   $|T_{11}|^2=\sin^2(\theta/2)$",
             ha="center", fontsize=12, color=CYAN)
    ax = fig.add_axes([0.07, 0.15, 0.55, 0.62])
    ax.set_xlim(0, 360); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("phase θ [°]"); ax.set_ylabel("output power")
    l1, = ax.plot([], [], color=CYAN, lw=2.2, label="port 1")
    l2, = ax.plot([], [], color=VIOLET, lw=2.2, label="port 2")
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    bx = fig.add_axes([0.70, 0.15, 0.27, 0.62])
    bars = bx.bar(["port 1", "port 2"], [0, 0], color=[CYAN, VIOLET])
    bx.set_ylim(0, 1.05); bx.set_title("instantaneous output", fontsize=10, color=MUTED)
    unit = bx.text(0.5, -0.2, "", transform=bx.transAxes, ha="center", fontsize=9, color=GREEN)

    def update(i):
        k = i + 1
        l1.set_data(np.degrees(th[:k]), P1[:k]); l2.set_data(np.degrees(th[:k]), P2[:k])
        bars[0].set_height(P1[i]); bars[1].set_height(P2[i])
        U = mzi(th[i], 0.0)
        err = np.abs(U.conj().T @ U - np.eye(2)).max()
        unit.set_text(f"‖U†U − I‖∞ = {err:.1e}")
        return (l1, l2, *bars, unit)

    make_anim(fig, update, len(th), 14, "mzi.gif", dpi=72)


# ----------------------------------------------------------------- talbot
def fig_talbot():
    N, dx, p = 1024, 2e-6, 16e-6
    x = np.arange(N) * dx
    t = 0.5 * (1 + np.cos(2 * math.pi * x / p))
    u0 = np.tile(t, (N, 1)).astype(complex)
    zT = 2 * p * p / LAM
    zs = np.linspace(0.0, 1.25 * zT, 50)
    profiles, contrast = [], []
    xs = x[: 2 * 8 * 4]  # 4 periods (8 px each)
    for z in zs:
        I = np.abs(angular_spectrum(u0[:1], dx, LAM, z)) ** 2  # single row: uniform in y
        I = I[0]
        profiles.append(I[: 4 * 8] / I.max())
        contrast.append((I.max() - I.min()) / (I.max() + I.min()))
    fig = plt.figure(figsize=(12, 4.5))
    fig.text(0.5, 0.94, "Talbot self-imaging: the grating pattern recurs at fractional Talbot distances",
             ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.86, r"$z_T = 2p^2/\lambda$ = %.2f mm,   period p = 16 µm,   λ = 633 nm" % (zT * 1e3),
             ha="center", fontsize=12, color=CYAN)
    ax = fig.add_axes([0.07, 0.15, 0.52, 0.62])
    xp = np.arange(len(profiles[0])) * dx * 1e6
    ax.plot(xp, 0.5 * (1 + np.cos(2 * math.pi * xp / 16.0)), color=MUTED, ls="--", lw=1.2, label="input")
    line, = ax.plot(xp, profiles[0], color=CYAN, lw=2.2, marker="o", ms=3, label="I(x, z)")
    ax.set_ylim(-0.05, 1.12); ax.set_xlim(0, xp[-1])
    ax.set_xlabel("x [µm]"); ax.set_ylabel("normalized intensity")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cx = fig.add_axes([0.70, 0.15, 0.27, 0.62])
    cx.plot(zs / zT, contrast, color=MUTED, lw=1)
    cdot, = cx.plot([], [], "o", color=AMBER, ms=7)
    cx.set_xlabel("z / z_T"); cx.set_ylabel("contrast")
    cx.set_title("revivals at z = 0, z_T", fontsize=10, color=MUTED)
    cx.set_ylim(-0.05, 1.05)
    for s in ("top", "right"):
        cx.spines[s].set_visible(False)

    def update(i):
        line.set_ydata(profiles[i])
        cdot.set_data([zs[i] / zT], [contrast[i]])
        return line, cdot

    make_anim(fig, update, len(zs), 12, "talbot.gif", dpi=72)


# --------------------------------------------------------------- aperture
def fig_aperture():
    n, dx, a = 1023, 8e-6, 0.2e-3
    x = (np.arange(n) - (n - 1) / 2) * dx
    X, Y = np.meshgrid(x, x)
    mask = (X**2 + Y**2 <= a**2)
    u0 = mask.astype(complex)
    a_eff = math.sqrt(mask.sum() * dx * dx / math.pi)
    zs = np.linspace(0.005, 0.10, 36)
    num = np.array([abs(angular_spectrum(u0, dx, LAM, z)[n // 2, n // 2]) ** 2 for z in zs])
    zan = np.linspace(0.005, 0.10, 400)
    NF = a_eff**2 / (LAM * zan)
    ian = 4 * np.sin(math.pi * NF / 2) ** 2
    fig = plt.figure(figsize=(12, 4.4))
    fig.text(0.5, 0.94, "Circular aperture: on-axis intensity (Fresnel diffraction)",
             ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.86, r"$I(0,z)/I_0 = 4\,\sin^2(\pi N_F/2)$,   $N_F = a^2/(\lambda z)$",
             ha="center", fontsize=12, color=CYAN)
    ax = fig.add_axes([0.08, 0.15, 0.6, 0.62])
    ax.plot(zan * 1e3, ian, color=MUTED, ls="--", lw=1.4, label="analytic")
    pts, = ax.plot([], [], "o", color=CYAN, ms=6, label="numerical (optlab)")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("I(0, z) / I₀")
    ax.set_ylim(0, 4.3); ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    info = fig.add_axes([0.73, 0.15, 0.24, 0.62]); info.axis("off")
    info.text(0, 0.9, f"effective radius a_eff = {a_eff * 1e3:.3f} mm", fontsize=10, color=FG)
    info.text(0, 0.78, "sampling dx = 8 µm, N = 1023", fontsize=10, color=MUTED)
    info.text(0, 0.62, "N_F > 1: Fresnel zones", fontsize=10, color=MUTED)
    info.text(0, 0.48, "N_F = 1: maximum of 4 I₀", fontsize=10, color=MUTED)
    info.text(0, 0.34, "validity: z < N·dx²/λ = 103 mm", fontsize=10, color=GREEN)

    def update(i):
        pts.set_data(zs[: i + 1] * 1e3, num[: i + 1])
        return (pts,)

    make_anim(fig, update, len(zs), 12, "aperture.gif", dpi=72)


# -------------------------------------------------------------- d2nn flow
def fig_d2nn_flow():
    d = load_digits()
    img = d.images[np.where(d.target == 3)[0][0]] / 16.0
    u0 = R.encode_field(img)
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 2 * np.pi, (3, R.N, R.N))
    masks = R.build_masks()
    planes = [np.abs(u0) ** 2]
    u = u0
    for k in range(3):
        u = angular_spectrum(u, R.DX, R.LAM, R.DIST[k])
        planes.append(np.abs(u) ** 2)
        u = u * np.exp(1j * phases[k])
    u = angular_spectrum(u, R.DX, R.LAM, R.DIST[3])
    det = np.abs(u) ** 2
    energies = np.array([float(np.sum(m * det)) for m in masks])
    crop = slice(R.N // 2 - 40, R.N // 2 + 40)
    titles = ["input (digit 3)", "after propagating 18 mm", "after layer 1 + 18 mm",
              "after layer 2 + 18 mm", "detector  |f|²"]
    imgs = planes + [det]

    fig = plt.figure(figsize=(12, 4.6))
    fig.text(0.5, 0.94, "Phase diffractive network (D2NN): light is distributed among 10 regions",
             ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.86, "random phases: flow illustration only, not a trained result",
             ha="center", fontsize=10, color=AMBER)
    axes = []
    arts = []
    for k in range(5):
        ax = fig.add_axes([0.02 + k * 0.155, 0.36, 0.14, 0.42])
        arts.append(ax.imshow(imgs[k][crop, crop] / det.max(), cmap="magma", vmin=0, vmax=1))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[k], fontsize=8, color=MUTED)
        axes.append(ax)
    # detector regions drawn on the last image
    for kk, m in enumerate(masks):
        rr, cc = np.nonzero(m)
        if rr.size == 0:
            continue
        r0, c0 = rr.min() - (R.N // 2 - 40), cc.min() - (R.N // 2 - 40)
        axes[4].add_patch(Rectangle((c0, r0), cc.max() - cc.min() + 1, rr.max() - rr.min() + 1,
                                    fill=False, ec=AMBER, lw=0.8))
    bx = fig.add_axes([0.10, 0.07, 0.85, 0.2])
    bars = bx.bar([str(k) for k in range(10)], np.zeros(10), color=CYAN)
    bx.set_ylim(0, energies.max() * 1.2); bx.set_ylabel("energy", fontsize=9)
    bx.set_xlabel("region (class)", fontsize=9)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    def update(i):
        for k, art in enumerate(arts):
            art.set_alpha(1.0 if i >= k * 6 else 0.2)
        frac = min(1.0, max(0.0, (i - 30) / 12))
        for kk, b in enumerate(bars):
            b.set_height(energies[kk] * frac)
        return tuple(arts) + tuple(bars)

    make_anim(fig, update, 48, 10, "d2nn_flow.gif", dpi=72)


# ------------------------------------------------------ results and verdicts
def fig_results():
    summary_path = RESULTS / "d2nn_digits_summary.json"
    if not summary_path.exists():
        print("d2nn_results.gif skipped: missing experiments/results/d2nn_digits_summary.json")
        return
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    seeds = [f["test_acc"] for f in s["final"]]
    labels = [f"seed {f['seed']}" for f in s["final"]]
    labels += ["D2NN mean", "logistic 64×64", "logistic 8×8", "C1 permuted\nlabels",
               "C2 untrained", "C3 phases = 0"]
    vals = seeds + [s["test_mean"], s["baselines"]["logistica_64x64"]["test_acc"],
                    s["baselines"]["logistica_8x8"]["test_acc"], s["controls"]["C1_permuted_mean"],
                    s["controls"]["C2_untrained_mean"], s["controls"]["C3_zero_phase"]]
    colors = [CYAN] * 5 + [VIOLET] + [MUTED] * 2 + [RED] * 3
    fig = plt.figure(figsize=(12, 4.6))
    fig.text(0.5, 0.94, "Handwritten digits (TEST): D2NN versus baselines and controls",
             ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.86, "CONTRATO-003 · TEST used once only · thresholds fixed in advance",
             ha="center", fontsize=10, color=CYAN)
    ax = fig.add_axes([0.06, 0.2, 0.92, 0.6])
    bars = ax.bar(range(len(vals)), np.zeros(len(vals)), color=colors)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy on TEST")
    ax.axhline(0.85, color=GREEN, ls="--", lw=1); ax.text(len(vals) - 0.5, 0.86, "H1 ≥ 0.85", color=GREEN,
                                                          ha="right", fontsize=8)
    ax.axhline(0.10, color=MUTED, ls=":", lw=1); ax.text(len(vals) - 0.5, 0.115, "chance 0.10", color=MUTED,
                                                         ha="right", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    vt = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center", color=FG, fontsize=10)
    seed_accs = np.array([f["test_acc"] for f in s["final"]])
    sd_seeds = float(seed_accs.std(ddof=1))
    n_bars = len(vals)

    def update(i):
        for k, b in enumerate(bars):
            b.set_height(vals[k] * min(1.0, max(0.0, (i - k * 2) / 8)))
        vt.set_text(f"D2NN mean {s['test_mean']:.3f} ± {sd_seeds:.3f} SD over 5 seeds   logistic 64×64 {vals[6]:.3f}")
        return tuple(bars)

    make_anim(fig, update, n_bars * 2 + 12, 12, "d2nn_results.gif", dpi=72)


def fig_verdicts():
    v2 = json.loads((RESULTS / "validation_summary.json").read_text(encoding="utf-8"))["verdicts"]
    d3p = RESULTS / "d2nn_digits_summary.json"
    v3 = json.loads(d3p.read_text(encoding="utf-8"))["verdicts"] if d3p.exists() else {}
    d4p = RESULTS / "regime_d2nn_summary.json"
    s4 = json.loads(d4p.read_text(encoding="utf-8")) if d4p.exists() else {}
    v4 = {code: s4.get(code, {}).get("verdict") for code in ("T1", "T2", "T3")}
    rows = [("CONTRATO-002", "E1", "Gaussian beam and Gouy phase", v2.get("E1")),
            ("CONTRATO-002", "E2", "Talbot self-imaging", v2.get("E2")),
            ("CONTRATO-002", "E3", "circular aperture", v2.get("E3")),
            ("CONTRATO-003", "H1", "D2NN ≥ 0.85 on TEST", v3.get("H1")),
            ("CONTRATO-003", "H2", "D2NN > logistic 64×64 (McNemar)", v3.get("H2")),
            ("CONTRATO-003", "C1", "permuted labels ≈ chance", v3.get("C1")),
            ("CONTRATO-003", "C2", "untrained ≈ chance", v3.get("C2")),
            ("CONTRATO-003", "C3", "no phases ≈ chance", v3.get("C3")),
            ("CONTRATO-004", "T1", "tilted plane waves (analytic)", v4["T1"]),
            ("CONTRATO-004", "T2", "zero-padding N = 128 vs 256 at 18 mm", v4["T2"]),
            ("CONTRATO-004", "T3", "pixel convergence, D2NN input", v4["T3"])]
    fig = plt.figure(figsize=(12, 4.6))
    fig.text(0.5, 0.94, "Preregistered verdicts", ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.86, "each criterion is checked against its threshold; failures are published too",
             ha="center", fontsize=10, color=MUTED)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.78]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, len(rows) + 0.5)
    ax.text(0.02, len(rows) + 0.1, "contract", color=MUTED, fontsize=10, weight="bold")
    ax.text(0.22, len(rows) + 0.1, "criterion", color=MUTED, fontsize=10, weight="bold")
    ax.text(0.80, len(rows) + 0.1, "verdict", color=MUTED, fontsize=10, weight="bold")
    marks = []
    for k, (c, code, desc, verdict) in enumerate(rows):
        y = len(rows) - k - 0.5
        ax.add_patch(Rectangle((0.0, y - 0.38), 1.0, 0.8, fc=PANEL, ec=BORDER, lw=0.6))
        ax.text(0.02, y, c, color=MUTED, fontsize=9, va="center")
        ax.text(0.22, y, f"{code}  {desc}", color=FG, fontsize=10, va="center")
        t = ax.text(0.80, y, "", color=FG, fontsize=10, va="center", weight="bold")
        marks.append((t, verdict_label(verdict)))

    def update(i):
        for k, (t, verdict) in enumerate(marks):
            if i >= k * 3:
                ok = verdict == "pass"
                t.set_text(("✔ " if ok else "✖ " if verdict == "fail" else "· ") + verdict)
                t.set_color(GREEN if ok else RED if verdict == "fail" else MUTED)
        return tuple(t for t, _ in marks)

    make_anim(fig, update, len(rows) * 3 + 6, 6, "verdicts.gif", dpi=72)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_header()
    fig_pipeline()
    fig_propagation()
    fig_fresnel()
    fig_mzi()
    fig_talbot()
    fig_aperture()
    fig_d2nn_flow()
    fig_results()
    fig_verdicts()


if __name__ == "__main__":
    main()
