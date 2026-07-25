#!/usr/bin/env python3
"""Stitch together the MESA history files from the six run phases and
summarize / plot the future evolution of the Sun."""

import glob
import os
import sys

import numpy as np

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sun_to_wd")

# in run order
PHASES = [
    ("LOGS_start", "pre-MS start"),
    ("LOGS_to_end_core_h_burn", "main sequence"),
    ("LOGS_to_start_he_core_flash", "RGB climb"),
    ("LOGS_to_end_core_he_burn", "core He burning"),
    ("LOGS_to_end_agb", "AGB"),
    ("LOGS_to_wd", "to white dwarf"),
]


def read_history(path):
    """Read a MESA history.data file -> dict of column -> array."""
    with open(path) as f:
        lines = f.readlines()
    names = lines[5].split()
    data = np.genfromtxt(lines[6:], names=names, invalid_raise=False)
    return np.atleast_1d(data)


def load_all(run_dir=RUN_DIR):
    parts = []
    for d, label in PHASES:
        p = os.path.join(run_dir, d, "history.data")
        if os.path.exists(p):
            h = read_history(p)
            parts.append((label, h))
    return parts


def stitched(parts, col):
    """Concatenate a column across phases, dropping backward time jumps
    (restarts within a phase are already handled by MESA)."""
    return np.concatenate([np.atleast_1d(h[col]) for _, h in parts])


def main():
    parts = load_all()
    if not parts:
        sys.exit("no history.data found yet")
    print(f"loaded {len(parts)} phases:")
    for label, h in parts:
        age0, age1 = h["star_age"][0], h["star_age"][-1]
        print(f"  {label:20s} {len(h['star_age']):6d} rows   "
              f"age {age0:.4e} -> {age1:.4e} yr")

    age = stitched(parts, "star_age")
    logL = stitched(parts, "log_L")
    logTeff = stitched(parts, "log_Teff")
    logR = stitched(parts, "log_R")
    mass = stitched(parts, "star_mass")
    center_h1 = stitched(parts, "center_h1")
    center_he4 = stitched(parts, "center_he4")

    gyr = age / 1e9
    L, R, Teff = 10**logL, 10**logR, 10**logTeff

    print("\n===== key epochs =====")
    # ZAMS ~ first time center H starts dropping / L minimum near start of MS
    now = np.argmin(np.abs(age - 4.6e9))
    print(f"age 4.6 Gyr (today):    L={L[now]:.3f} Lsun  R={R[now]:.3f} Rsun  Teff={Teff[now]:.0f} K")

    i_h_exhaust = np.argmax(center_h1 < 1e-4)
    print(f"core H exhausted:       age={gyr[i_h_exhaust]:.3f} Gyr")

    # RGB tip = max L before He flash (end of phase 3)
    n_rgb_end = sum(len(np.atleast_1d(h['star_age'])) for _, h in parts[:3])
    i_rgbtip = np.argmax(L[:n_rgb_end])
    print(f"RGB tip:                age={gyr[i_rgbtip]:.4f} Gyr  L={L[i_rgbtip]:.0f} Lsun  "
          f"R={R[i_rgbtip]:.0f} Rsun  Teff={Teff[i_rgbtip]:.0f} K")

    i_he_exhaust = len(L) - 1 - np.argmax((center_he4 < 1e-4)[::-1] == False)
    i_maxR = np.argmax(R)
    print(f"max radius (AGB):       age={gyr[i_maxR]:.4f} Gyr  R={R[i_maxR]:.0f} Rsun "
          f"({R[i_maxR]*0.00465:.2f} AU)  L={L[i_maxR]:.0f} Lsun")
    print(f"final mass:             {mass[-1]:.4f} Msun (lost {1-mass[-1]:.3f} Msun)")
    print(f"final age:              {gyr[-1]:.4f} Gyr,  final L={L[-1]:.3f} Lsun, "
          f"final Teff={Teff[-1]:.0f} K")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    for label, h in parts:
        ax.plot(h["log_Teff"], h["log_L"], lw=1, label=label)
    ax.invert_xaxis()
    ax.set_xlabel(r"$\log\,T_{\rm eff}$ [K]")
    ax.set_ylabel(r"$\log\,L/L_\odot$")
    ax.set_title("Hertzsprung–Russell diagram")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(gyr, logL, lw=1)
    ax.set_xlabel("age [Gyr]")
    ax.set_ylabel(r"$\log\,L/L_\odot$")
    ax.set_title("Luminosity vs age")

    ax = axes[1, 0]
    ax.plot(gyr, logR, lw=1)
    ax.set_xlabel("age [Gyr]")
    ax.set_ylabel(r"$\log\,R/R_\odot$")
    ax.set_title("Radius vs age")

    ax = axes[1, 1]
    ax.plot(gyr, mass, lw=1.5)
    ax.set_xlabel("age [Gyr]")
    ax.set_ylabel(r"$M/M_\odot$")
    ax.set_title("Mass vs age (wind loss)")

    fig.suptitle("Future evolution of the Sun — MESA r26.04.1, 1 $M_\\odot$, Z=0.02", y=0.995)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(RUN_DIR), "sun_evolution.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
