#!/usr/bin/env python3
"""Benchmark WHFast configurations for the 7 Gyr 'hero run'."""
import time

import rebound
import reboundx

# J2000-ish elements, good enough for benchmarking
PLANETS = [
    # name, m (Msun), a (AU), e
    ("Mercury", 1.660e-7, 0.3871, 0.2056),
    ("Venus",   2.448e-6, 0.7233, 0.0068),
    ("Earth",   3.003e-6, 1.0000, 0.0167),
    ("Mars",    3.227e-7, 1.5237, 0.0934),
    ("Jupiter", 9.545e-4, 5.2044, 0.0489),
    ("Saturn",  2.858e-4, 9.5826, 0.0565),
    ("Uranus",  4.366e-5, 19.2185, 0.0457),
    ("Neptune", 5.151e-5, 30.1104, 0.0113),
]


def build(skip_mercury=False):
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0)
    for i, (name, m, a, e) in enumerate(PLANETS):
        if skip_mercury and name == "Mercury":
            continue
        sim.add(m=m, a=a, e=e, f=i * 0.7)  # spread anomalies
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0       # no per-step synchronization
    sim.ri_whfast.corrector = 17       # high-order symplectic corrector
    return sim


def bench(label, sim, dt_days, t_end_yr=2e5):
    sim.dt = dt_days / 365.25
    n_steps = int(t_end_yr / sim.dt)
    t0 = time.perf_counter()
    sim.integrate(t_end_yr, exact_finish_time=0)
    wall = time.perf_counter() - t0
    rate = n_steps / wall
    yr_per_s = t_end_yr / wall
    days_7gyr = 7e9 / yr_per_s / 86400
    print(f"{label:34s} dt={dt_days:5.1f} d  {rate/1e6:6.2f} Msteps/s  "
          f"{yr_per_s/1e6:7.3f} Myr/s  -> 7 Gyr in {days_7gyr:6.2f} days")


print(f"rebound {rebound.__version__}; single core benchmark, 2e5 yr segments")

s = build();               bench("8 planets", s, 4.0)
s = build();               bench("8 planets, dt=8d (P_merc/11)", s, 8.0)
s = build(True);           bench("7 planets (no Mercury)", s, 18.0)

s = build()
rx = reboundx.Extras(s)
gr = rx.load_force("gr_potential")
gr.params["c"] = 63241.077  # AU/yr
rx.add_force(gr)
bench("8 planets + gr_potential", s, 4.0)

s = build(True)
rx = reboundx.Extras(s)
gr = rx.load_force("gr_potential")
gr.params["c"] = 63241.077
rx.add_force(gr)
bench("7 planets + gr_potential", s, 18.0)
