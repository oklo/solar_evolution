#!/usr/bin/env python3
"""Cross-check: REBOUNDx constant-time-lag tide (Zahn-calibrated per chunk)
vs the secular ODE, over the last ~14 Myr before the RGB tip where the tidal
term dominates. Inner four planets; initial conditions taken from the secular
solution at the segment start so both methods integrate the same problem.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from secular_tides import Star, load_star, integrate_planet, PLANETS, RSUN, AU
from rebound_driver import build_sim, update_star, remove_engulfed
import rebound

T_SEG0 = 12.265e9
T_SEG1 = 12.279e9
RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "sun_to_wd")

star = Star(load_star(RUN_DIR))

# --- secular reference, from today to segment end ---
print("=== secular ===")
sec = {}
for name, m_p, a0 in PLANETS:
    t, a, t_eng = integrate_planet(star, m_p, a0, 4.61e9, T_SEG1)
    a_seg0 = np.interp(T_SEG0, t, a)
    sec[name] = dict(a_seg0=a_seg0, t_eng=t_eng,
                     a_end=a[-1] if t_eng is None else None)
    print(f"{name:8s} a(seg0)={a_seg0:.4f}  "
          + (f"engulfed {t_eng/1e9:.5f} Gyr" if t_eng else f"a(end)={a[-1]:.4f}"))

# --- rebound segment with same ICs ---
print("=== rebound (Zahn-calibrated CTL tide) ===")
sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")
sim.add(m=float(star.M(T_SEG0)), hash="Sun")
for i, (name, m_p, a0) in enumerate(PLANETS):
    if sec[name]["t_eng"] and sec[name]["t_eng"] < T_SEG0:
        continue
    sim.add(m=m_p, a=float(sec[name]["a_seg0"]), e=0.0, f=i * 1.7, hash=name)
sim.move_to_com()
sim.integrator = "whfast"
sim.ri_whfast.safe_mode = 0
sim.dt = 4.0 / 365.25

import reboundx
rx = reboundx.Extras(sim)
mod = rx.load_operator("modify_orbits_direct")
rx.add_operator(mod)   # Zahn tide as per-planet tau_a; no GR needed here

update_star(sim, star, T_SEG0)
log, events = [], []
t = T_SEG0
t_wall = time.perf_counter()
while t < T_SEG1 and sim.N > 1:
    chunk = 2e3  # yr; R changes fast near the tip
    t_next = min(t + chunk, T_SEG1)
    sim.integrate(sim.t + (t_next - t), exact_finish_time=0)
    t = T_SEG0 + sim.t
    R_au = update_star(sim, star, t)
    for h, a in remove_engulfed(sim, R_au, log):
        events.append((t, a))
        print(f"engulfed at t={t/1e9:.5f} Gyr (a={a:.4f}, R={R_au:.4f})",
              flush=True)
print(f"wall {time.perf_counter()-t_wall:.1f} s")
for p in sim.particles[1:]:
    o = p.orbit(primary=sim.particles[0])
    print(f"survivor: a={o.a:.4f} AU e={o.e:.4f}")
