#!/usr/bin/env python3
"""Re-integrate the hero run's tail (post-Venus-engulfment onward) with
MERCURIUS, which handles close encounters that broke WHFast when Earth and
Mars's orbits began to cross. Starts from a healthy archive snapshot, keeps
mass loss + GR; tides are negligible in this era (rate < 1e-15/yr) and the
tide operator is left off.
"""

import os
import sys
import warnings

import numpy as np
import rebound
import reboundx

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from secular_tides import Star, load_star  # noqa: E402
from rebound_driver import update_star, remove_engulfed  # noqa: E402

T0_ABS = 10.75e9
SNAP_IDX = 3800          # 11.95291 Gyr: post-Venus, eccentricities still ~0.02
T_END_ABS = 12.4e9

star = Star(load_star(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "sun_to_wd_calib")))
sa = rebound.Simulationarchive("hero.bin")
sim = sa[SNAP_IDX]
t_start = T0_ABS + sim.t
print(f"resuming from snapshot {SNAP_IDX}: t={t_start/1e9:.5f} Gyr, "
      f"N={sim.N-1} planets", flush=True)

sim.integrator = "mercurius"
sim.dt = 12.0 / 365.25          # P_Earth/40 for encounter fidelity
sim.ri_mercurius.r_crit_hill = 4.0

rx = reboundx.Extras(sim)
gr = rx.load_force("gr_potential")
gr.params["c"] = 63241.077
rx.add_force(gr)

ARCHIVE = "hero_tail.bin"
if os.path.exists(ARCHIVE):
    os.remove(ARCHIVE)

t = t_start
n_chunk = 0
while t < T_END_ABS and sim.N > 1:
    M = star.M(t)
    dM = np.abs(star.dMdt(t)) + 1e-30
    chunk = np.clip(1e-4 * M / dM, 1e3, 1e6)
    t_next = min(t + chunk, T_END_ABS)
    sim.integrate(sim.t + (t_next - t), exact_finish_time=0)
    t = T0_ABS + sim.t
    R_au, _ = update_star(sim, star, t, tides_on=False)
    log = []
    for h, a in remove_engulfed(sim, R_au, log):
        print(f"t={t/1e9:.5f} Gyr: engulfed/removed at a={a:.4f} AU",
              flush=True)
    # collision/ejection sanity: report any hyperbolic or NaN orbit
    for p in sim.particles[1:]:
        o = p.orbit(primary=sim.particles[0])
        if not np.isfinite(o.a):
            print(f"t={t/1e9:.5f} Gyr: NON-FINITE ORBIT — stopping",
                  flush=True)
            sys.exit(1)
    sim.save_to_file(ARCHIVE, delete_file=False)
    n_chunk += 1
    if n_chunk % 100 == 0:
        oE = sim.particles[1].orbit(primary=sim.particles[0])
        oM = sim.particles[2].orbit(primary=sim.particles[0])
        print(f"t={t/1e9:.5f} Gyr  e_E={oE.e:.3f} a_E={oE.a:.3f}  "
              f"e_M={oM.e:.3f} a_M={oM.a:.3f}", flush=True)

print(f"tail done at t={t/1e9:.5f} Gyr with {sim.N-1} planets", flush=True)
for p in sim.particles[1:]:
    o = p.orbit(primary=sim.particles[0])
    print(f"  a={o.a:.4f} AU e={o.e:.4f}", flush=True)
