#!/usr/bin/env python3
"""Secular orbital evolution of the inner planets under solar mass loss and
convective (Zahn) equilibrium tides, driven by a MESA history.

da/dt = a * [ -Mdot_tot/M_tot ]                                (isotropic wind)
      - 6*(2/21)*f' * (M_env/M) / tau_conv * q(1+q) * (R/a)^8 * a   (Zahn tide)

with tau_conv = (M_env R^2 / L)^(1/3) and the Goldreich-Nicholson style
reduction f' = min[1, (P_orb / 2 tau_conv)^2].

Usage: secular_tides.py [run_dir]   (default: ../sun_to_wd, stitched phases)
"""

import glob
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

MSUN = 1.989e30      # kg
RSUN = 6.957e8       # m
LSUN = 3.828e26      # W
AU = 1.496e11        # m
YR = 3.156e7         # s

PLANETS = [
    ("Mercury", 1.660e-7, 0.3871),
    ("Venus",   2.448e-6, 0.7233),
    ("Earth",   3.003e-6, 1.0000),
    ("Mars",    3.227e-7, 1.5237),
]

PHASE_ORDER = ["LOGS_start", "LOGS_to_end_core_h_burn", "LOGS_to_start_he_core_flash",
               "LOGS_to_end_core_he_burn", "LOGS_to_end_agb", "LOGS_to_wd",
               "LOGS"]  # plain LOGS last, for single-phase dirs


def read_history(path):
    with open(path) as f:
        lines = f.readlines()
    return np.atleast_1d(np.genfromtxt(lines[6:], names=lines[5].split(),
                                       invalid_raise=False))


def load_star(run_dir):
    cols = ["star_age", "star_mass", "log_L", "log_R", "he_core_mass"]
    data = {c: [] for c in cols}
    for d in PHASE_ORDER:
        p = os.path.join(run_dir, d, "history.data")
        if not os.path.exists(p):
            continue
        h = read_history(p)
        for c in cols:
            data[c].append(np.atleast_1d(h[c]))
    if not data["star_age"]:
        sys.exit(f"no history.data under {run_dir}")
    arr = {c: np.concatenate(v) for c, v in data.items()}
    # enforce strictly increasing age (drop tiny backsteps at phase seams)
    age = arr["star_age"]
    keep = np.concatenate([[True], np.diff(age) > 0])
    while not keep.all():
        for c in arr:
            arr[c] = arr[c][keep]
        age = arr["star_age"]
        keep = np.concatenate([[True], np.diff(age) > 0])
    return arr


class Star:
    def __init__(self, arr):
        t = arr["star_age"]
        self.t0, self.t1 = t[0], t[-1]
        self.M = PchipInterpolator(t, arr["star_mass"])          # Msun
        self.logR = PchipInterpolator(t, arr["log_R"])
        self.logL = PchipInterpolator(t, arr["log_L"])
        self.Menv = PchipInterpolator(
            t, np.maximum(arr["star_mass"] - arr["he_core_mass"], 1e-6))
        self.dMdt = self.M.derivative()                          # Msun/yr

    def tau_conv_yr(self, t):
        Menv = self.Menv(t) * MSUN
        R = 10 ** self.logR(t) * RSUN
        L = 10 ** self.logL(t) * LSUN
        return (Menv * R * R / L) ** (1.0 / 3.0) / YR


def integrate_planet(star, m_p, a0_au, t_start, t_end):
    """Integrate a(t) in AU; time in yr. Returns t_grid, a(t), t_engulf."""

    def rhs(t, y):
        a = y[0]
        M = star.M(t)
        q = m_p / M
        R_au = 10 ** star.logR(t) * RSUN / AU
        tau = star.tau_conv_yr(t)
        P_orb = np.sqrt(a ** 3 / M)                  # yr (G=4pi^2 units)
        fprime = min(1.0, (P_orb / (2.0 * tau)) ** 2)
        wind = -star.dMdt(t) / (M + m_p) * a         # dM/dt<0 -> da/dt>0
        tide = -(12.0 / 21.0) * fprime * (star.Menv(t) / M) / tau \
            * q * (1.0 + q) * (R_au / a) ** 8 * a
        return [wind + tide]

    def engulfed(t, y):
        return y[0] - 10 ** star.logR(t) * RSUN / AU
    engulfed.terminal = True
    engulfed.direction = -1

    sol = solve_ivp(rhs, (t_start, t_end), [a0_au], method="LSODA",
                    rtol=1e-8, atol=1e-12, events=engulfed, dense_output=True,
                    max_step=(t_end - t_start) / 200.0)
    t_eng = sol.t_events[0][0] if len(sol.t_events[0]) else None
    return sol.t, sol.y[0], t_eng


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sun_to_wd")
    arr = load_star(run_dir)
    star = Star(arr)
    t_start, t_end = 4.61e9, min(star.t1, 12.5e9)
    print(f"star history {star.t0:.3e} -> {star.t1:.3e} yr; "
          f"integrating {t_start:.3e} -> {t_end:.3e}")

    results = {}
    for name, m_p, a0 in PLANETS:
        t, a, t_eng = integrate_planet(star, m_p, a0, t_start, t_end)
        results[name] = (t, a, t_eng)
        if t_eng:
            print(f"{name:8s} ENGULFED at t = {t_eng/1e9:.4f} Gyr "
                  f"(a = {a[-1]:.4f} AU)")
        else:
            print(f"{name:8s} survives; final a = {a[-1]:.4f} AU "
                  f"(started {a0:.4f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    tt = np.linspace(t_start, t_end, 4000)
    ax.plot(tt / 1e9, 10 ** star.logR(tt) * RSUN / AU, "k-", lw=2,
            label=r"$R_\star$")
    for name, (t, a, t_eng) in results.items():
        ax.plot(t / 1e9, a, lw=1.5, label=name)
        if t_eng:
            ax.plot(t_eng / 1e9, np.interp(t_eng, t, a), "rx", ms=8)
    ax.set_xlabel("age [Gyr]")
    ax.set_ylabel("distance [AU]")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Solar radius vs planetary semi-major axes (secular model)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "secular_orbits.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
