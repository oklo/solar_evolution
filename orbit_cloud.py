#!/usr/bin/env python3
"""Osculating-orbit point clouds from a REBOUND SimulationArchive, following
the ../solar_system workflow: each surviving planet's instantaneous ellipse is
sampled at one-day mean-anomaly cadence (first point at perihelion), so dot
spacing encodes orbital speed; a periastron guide segment fades in with e.
"""

import numpy as np
import rebound


def solve_kepler(M, e):
    M = np.mod(M, 2.0 * np.pi)
    E = M.copy()
    for _ in range(30):
        delta = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= delta
        if np.max(np.abs(delta)) < 1e-12:
            break
    return E


def rotation_matrix(inc, Omega, omega):
    ci, si = np.cos(inc), np.sin(inc)
    cO, sO = np.cos(Omega), np.sin(Omega)
    co, so = np.cos(omega), np.sin(omega)
    return np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si, co * si, ci],
    ])


def orbit_cloud_xy(orbit, sun_xy, max_pts=800):
    """Dots on the osculating ellipse at one-day mean-anomaly cadence,
    heliocentric, projected on the ecliptic. First point = perihelion."""
    P_days = orbit.P * 365.25
    count = int(np.clip(round(abs(P_days)), 60, max_pts))
    M = np.arange(count) * (2.0 * np.pi / count)
    E = solve_kepler(M, orbit.e)
    x = orbit.a * (np.cos(E) - orbit.e)
    y = orbit.a * np.sqrt(max(0.0, 1.0 - orbit.e ** 2)) * np.sin(E)
    pts = np.column_stack([x, y, np.zeros_like(x)])
    R = rotation_matrix(orbit.inc, orbit.Omega, orbit.omega)
    pts = pts @ R.T
    return pts[:, :2] + np.asarray(sun_xy)


def periastron_seg_xy(orbit, sun_xy, length):
    R = rotation_matrix(orbit.inc, orbit.Omega, orbit.omega)
    d = (R @ np.array([1.0, 0.0, 0.0]))[:2]
    n = np.linalg.norm(d)
    if n == 0:
        return None
    d = d / n * length
    s = np.asarray(sun_xy)
    return np.array([s, s + d])


class ArchiveOrbits:
    """Time-indexed access to per-planet osculating state from one or more
    archives. `path` may be a single filename or a ';'-separated list; later
    archives supersede earlier ones where their time ranges overlap (used to
    splice the MERCURIUS-re-integrated tail over the WHFast head).

    times are ABSOLUTE stellar ages: t_abs = t0_abs + sim.t
    """

    def __init__(self, path, t0_abs):
        self.t0_abs = t0_abs
        self.sas, self.idx, times = [], [], []
        for k, p in enumerate(str(path).split(";")):
            sa = rebound.Simulationarchive(p.strip())
            self.sas.append(sa)
            for j, s in enumerate(sa):
                times.append(s.t + t0_abs)
                self.idx.append((k, j))
        times = np.array(times)
        # later archives win on overlap: drop earlier-archive entries at or
        # beyond the next archive's start
        if len(self.sas) > 1:
            starts = [self.sas[k][0].t + t0_abs for k in range(len(self.sas))]
            keep = np.array([not any(times[m] >= starts[k2]
                                     for k2 in range(self.idx[m][0] + 1,
                                                     len(self.sas)))
                             for m in range(len(times))])
            times = times[keep]
            self.idx = [x for x, kp in zip(self.idx, keep) if kp]
        order = np.argsort(times)
        self.times = times[order]
        self.idx = [self.idx[o] for o in order]

    def snapshot(self, t_abs):
        i = int(np.clip(np.searchsorted(self.times, t_abs), 0,
                        len(self.times) - 1))
        k, j = self.idx[i]
        sim = self.sas[k][j]
        sun = sim.particles[0]
        out = []
        for p in sim.particles[1:]:
            o = p.orbit(primary=sun)
            name = None
            h = p.hash.value if hasattr(p.hash, "value") else int(p.hash)
            for cand in ("Mercury", "Venus", "Earth", "Mars",
                         "Jupiter", "Saturn", "Uranus", "Neptune"):
                hc = rebound.hash(cand)
                hc = hc.value if hasattr(hc, "value") else int(hc)
                if h == hc:
                    name = cand
                    break
            out.append(dict(name=name, orbit=o,
                            xy=(p.x - sun.x, p.y - sun.y)))
        return self.times[i], out
