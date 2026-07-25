#!/usr/bin/env python3
"""WHFast driver for the solar-system 'hero run': REBOUND + REBOUNDx
(gr_potential + tides_constant_time_lag), with the star's mass, radius and
tidal time-lag updated each chunk from MESA history interpolants.

The constant-time-lag tide is calibrated to Zahn each chunk:
  Hut/CTL (e=0, Omega=0):  adot/a = -6 k2 tau n q(1+q) (R/a)^8
  Zahn convective:         adot/a = -(12/21) f' (Menv/M)/tau_c q(1+q) (R/a)^8
  =>  k2*tau = (2/21) f' (Menv/M) / (tau_c * n)
We fix k2 = 0.03 and fold the rest into tau. Time lag is set per-chunk using
the innermost surviving planet's mean motion (the (R/a)^8 factor makes the
innermost planet utterly dominant, so this approximation is excellent).

Usage:
  rebound_driver.py --run-dir ../sun_to_wd --t0 4.61e9 --t1 12.4e9 \
      --archive hero.bin [--interval 1e6] [--test-segment]
"""

import argparse
import os
import sys

import numpy as np
import rebound
import reboundx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from secular_tides import AU, RSUN, YR, Star, load_star  # noqa: E402

K2 = 0.03
PLANETS = [
    ("Mercury", 1.660e-7, 0.3871, 0.2056),
    ("Venus",   2.448e-6, 0.7233, 0.0068),
    ("Earth",   3.003e-6, 1.0000, 0.0167),
    ("Mars",    3.227e-7, 1.5237, 0.0934),
    ("Jupiter", 9.545e-4, 5.2044, 0.0489),
    ("Saturn",  2.858e-4, 9.5826, 0.0565),
    ("Uranus",  4.366e-5, 19.2185, 0.0457),
    ("Neptune", 5.151e-5, 30.1104, 0.0113),
]


def build_sim(star, t0, dt_days, seed=0):
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    M0 = float(star.M(4.61e9))
    Mt = float(star.M(t0))
    sim.add(m=Mt, hash="Sun")
    rng = np.random.default_rng(seed)
    for name, m, a, e in PLANETS:
        # adiabatic orbit expansion for the (tiny) mass already lost by t0
        a_t = a * (M0 + m) / (Mt + m)
        sim.add(m=m, a=a_t, e=e, f=float(rng.uniform(0, 2 * np.pi)), hash=name)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.dt = dt_days / 365.25

    rx = reboundx.Extras(sim)
    gr = rx.load_force("gr_potential")   # position-only: WHFast-safe
    gr.params["c"] = 63241.077
    rx.add_force(gr)
    # tide operator is NOT attached here: it costs ~7x the WHFast kernel and
    # does nothing until the star approaches the RGB tip. main() attaches it
    # when the tidal rate becomes dynamically relevant.
    mod = rx.load_operator("modify_orbits_direct")
    return sim, rx, mod


def update_star(sim, star, t_abs, tides_on=True):
    """Set stellar mass/radius and per-planet Zahn tidal decay rates at age t.

    Mass loss acts through the mass update itself (adiabatic a ~ 1/M emerges
    from many small chunks); the tide is imposed as modify_orbits_direct
    tau_a = a/adot from Zahn's convective equilibrium tide.
    Returns (R_au, max_rate) so the caller can decide when to attach the
    tide operator and how to tune dt.
    """
    sun = sim.particles["Sun"]
    M = float(star.M(t_abs))
    sun.m = M
    R_au = float(10 ** star.logR(t_abs)) * RSUN / AU
    sun.r = R_au

    tau_c = star.tau_conv_yr(t_abs)
    Menv = float(star.Menv(t_abs))
    max_rate = 0.0
    for p in sim.particles[1:]:
        o = p.orbit(primary=sun)
        q = p.m / M
        P_orb = np.sqrt(o.a ** 3 / M)
        fprime = min(1.0, (P_orb / (2.0 * tau_c)) ** 2)
        rate = (12.0 / 21.0) * fprime * (Menv / M) / tau_c \
            * q * (1.0 + q) * (R_au / o.a) ** 8          # -adot/a, 1/yr
        max_rate = max(max_rate, rate)
        if tides_on:
            p.params["tau_a"] = -1.0 / rate if rate > 1e-30 else -1e30
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1
    return R_au, max_rate


def remove_engulfed(sim, R_au, log):
    removed = []
    sun = sim.particles["Sun"]
    for p in list(sim.particles[1:]):
        o = p.orbit(primary=sun)
        if o.a * (1 - o.e) < R_au:
            removed.append((p.hash, o.a))
            log.append(f"ENGULFED a={o.a:.4f} AU")
            sim.remove(hash=p.hash)
    if removed:
        sim.move_to_com()
        sim.ri_whfast.recalculate_coordinates_this_timestep = 1
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "sun_to_wd"))
    ap.add_argument("--t0", type=float, default=10.75e9)
    ap.add_argument("--t1", type=float, default=12.4e9)
    ap.add_argument("--dt-days", type=float, default=4.0)
    ap.add_argument("--archive", default="hero.bin")
    ap.add_argument("--interval", type=float, default=1e6,
                    help="archive snapshot interval [yr]")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    star = Star(load_star(args.run_dir))
    sim, rx, tide_op = build_sim(star, args.t0, args.dt_days, args.seed)
    tides_on = False
    update_star(sim, star, args.t0, tides_on)

    if os.path.exists(args.archive):
        os.remove(args.archive)

    def retune_dt():
        """dt = P_inner/N: N=20 in the tidal era, N=10 on the quiet climb
        (R < 20 Rsun, where the inner system is dynamically cold)."""
        P_min = min(np.sqrt(p.orbit(primary=sim.particles[0]).a ** 3
                            / sim.particles[0].m)
                    for p in sim.particles[1:])
        sim.dt = P_min / (20.0 if tides_on else 10.0)
        sim.ri_whfast.recalculate_coordinates_this_timestep = 1

    retune_dt()

    log = []
    # chunking: params change slowly except near the RGB tip; adapt chunk so
    # that |dM|/M < 1e-4 and |dR|/R < 1% across it, within [1e3, 1e6] yr.
    # One archive snapshot per chunk -> adaptive output density for free.
    t = args.t0
    n_chunk = 0
    while t < args.t1 and sim.N > 1:
        M, R = star.M(t), 10 ** star.logR(t)
        dM = np.abs(star.dMdt(t)) + 1e-30
        dRdt = np.abs(star.logR.derivative()(t)) * R * np.log(10) + 1e-30
        chunk = np.clip(min(1e-4 * M / dM, 0.01 * R / dRdt), 1e3, 1e6)
        t_next = min(t + chunk, args.t1)
        sim.integrate(sim.t + (t_next - t), exact_finish_time=0)
        t = args.t0 + sim.t
        R_au, max_rate = update_star(sim, star, t, tides_on)
        if not tides_on and max_rate > 1e-12:
            tides_on = True
            rx.add_operator(tide_op)
            update_star(sim, star, t, tides_on)   # set tau_a before stepping
            retune_dt()
            print(f"t={t/1e9:.5f} Gyr: tides attached "
                  f"(max rate {max_rate:.2e}/yr), dt -> "
                  f"{sim.dt*365.25:.1f} d", flush=True)
        for h, a in remove_engulfed(sim, R_au, log):
            print(f"t={t/1e9:.5f} Gyr: engulfed planet at a={a:.4f} AU "
                  f"(R_star={R_au:.4f} AU)", flush=True)
            retune_dt()
            print(f"  dt -> {sim.dt*365.25:.1f} d", flush=True)
        sim.save_to_file(args.archive, delete_file=False)
        n_chunk += 1
        if n_chunk % 100 == 0:
            print(f"t={t/1e9:.5f} Gyr  dt={sim.dt*365.25:.1f} d  "
                  f"N={sim.N-1}  rate={max_rate:.2e}", flush=True)

    print(f"done at t={t/1e9:.5f} Gyr with {sim.N-1} planets remaining")
    for p in sim.particles[1:]:
        o = p.orbit(primary=sim.particles[0])
        print(f"  a={o.a:.4f} AU e={o.e:.4f}")


if __name__ == "__main__":
    main()
