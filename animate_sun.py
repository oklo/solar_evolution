#!/usr/bin/env python3
"""v1 talk animation, 16:9 (PowerPoint): the Sun's future.

Left: HR diagram. Full future track dim; the Sun traces it from the present
day (ZAMS->today already drawn), marker sized by log R, blackbody-colored.
Right: inner solar system, face-on, to scale in AU; static present-day orbits;
the Sun's disk grows/colors with R(t), Teff(t).

Pacing: uniform arc length in (dlogTeff, dlogL, dlogR) space -- equal visual
change per frame, so the He flash and AGB get their screen time and the MS
glides. Age + timestep readouts keep the time compression honest.
"""

import os

import numpy as np
from scipy.interpolate import PchipInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle

BASE = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get("SUN_RUN", os.path.join(BASE, "sun_to_wd"))
RSUN_AU = 6.957e8 / 1.496e11

# optional N-body archive for live osculating orbits on the right panel
ARCHIVE = os.environ.get("NBODY_ARCHIVE", "")
ARCHIVE_T0 = float(os.environ.get("NBODY_T0", "10.75e9"))

PHASES = [
    ("LOGS_to_end_core_h_burn", "Main sequence"),
    ("LOGS_to_start_he_core_flash", "Red giant branch"),
    ("LOGS_to_end_core_he_burn", "He flash → core He burning"),
    ("LOGS_to_end_agb", "Asymptotic giant branch"),
    ("LOGS_to_wd", "Envelope ejection → white dwarf"),
]

ORBITS = [  # name, a (AU), display color (muted, identity only)
    ("Mercury", 0.387, "#8a8f98"),
    ("Venus", 0.723, "#c9a86a"),
    ("Earth", 1.000, "#6aa9d8"),
    ("Mars", 1.524, "#c96a5a"),
]

# ---------- data ----------

def read_history(path):
    with open(path) as f:
        lines = f.readlines()
    return np.atleast_1d(np.genfromtxt(lines[6:], names=lines[5].split(),
                                       invalid_raise=False))


def load():
    age, lT, lL, lR, phase = [], [], [], [], []
    for i, (d, label) in enumerate(PHASES):
        h = read_history(os.path.join(RUN, d, "history.data"))
        age.append(h["star_age"]); lT.append(h["log_Teff"])
        lL.append(h["log_L"]); lR.append(h["log_R"])
        phase.append(np.full(len(np.atleast_1d(h["star_age"])), i))
    age = np.concatenate(age); lT = np.concatenate(lT)
    lL = np.concatenate(lL); lR = np.concatenate(lR)
    phase = np.concatenate(phase)
    keep = np.concatenate([[True], np.diff(age) > 0])
    return age[keep], lT[keep], lL[keep], lR[keep], phase[keep]


def resample_arclength(age, lT, lL, lR, phase, n_pre=90, n_main=810):
    """Return frame-wise arrays: n_pre frames ZAMS->today, n_main after."""
    # normalized structural-change metric
    ds = np.sqrt((np.diff(lT) / np.ptp(lT)) ** 2 +
                 (np.diff(lL) / np.ptp(lL)) ** 2 +
                 (np.diff(lR) / np.ptp(lR)) ** 2)
    s = np.concatenate([[0], np.cumsum(ds)])
    i_now = np.argmin(np.abs(age - 4.61e9))
    s_pre = np.linspace(0, s[i_now], n_pre, endpoint=False)
    s_main = np.linspace(s[i_now], s[-1], n_main)
    s_f = np.concatenate([s_pre, s_main])
    out = {}
    for k, v in dict(age=age, lT=lT, lL=lL, lR=lR).items():
        out[k] = PchipInterpolator(s, v)(s_f)
    out["phase"] = np.interp(s_f, s, phase)
    out["i_now"] = n_pre
    return out

# ---------- limb darkening (Claret-style quadratic, V band, solar Z) ----------
# I(mu)/I(1) = 1 - a(1-mu) - b(1-mu)^2 ; (a,b) vs Teff, approx from Claret
# tables (giants/dwarfs blended -- display-grade, not photometric-grade).
LD_TEFF = np.array([3000, 3500, 4000, 4500, 5000, 5800, 7000, 10000,
                    20000, 30000], float)
LD_A = np.array([0.86, 0.82, 0.75, 0.65, 0.56, 0.44, 0.35, 0.25,
                 0.15, 0.10])
LD_B = np.array([0.10, 0.08, 0.10, 0.17, 0.20, 0.26, 0.28, 0.30,
                 0.28, 0.25])


def ld_coeffs(T):
    return (np.interp(T, LD_TEFF, LD_A), np.interp(T, LD_TEFF, LD_B))


def sun_disk_rgba(T, npix=257):
    """RGBA image of the bare disk: hard limb, Claret quadratic darkening,
    limb reddening via T_local = Teff * I^(1/4)."""
    a, b = ld_coeffs(T)
    x = np.linspace(-1, 1, npix)
    xx, yy = np.meshgrid(x, x)
    r2 = xx * xx + yy * yy
    inside = r2 <= 1.0
    mu = np.zeros_like(r2)
    mu[inside] = np.sqrt(1.0 - r2[inside])
    I = np.clip(1.0 - a * (1.0 - mu) - b * (1.0 - mu) ** 2, 0.05, 1.0)
    img = np.zeros((npix, npix, 4))
    # local color from I as a T^4 proxy (gray-atmosphere limb reddening)
    T_loc = T * I ** 0.25
    # vectorize bb_rgb over a coarse T grid for speed
    T_grid = np.linspace(T_loc[inside].min(), T, 32)
    cols = np.array([bb_rgb(t) for t in T_grid])
    idx = np.searchsorted(T_grid, T_loc[inside]).clip(0, 31)
    img[inside, :3] = cols[idx] * I[inside, None] ** 0.6  # mild brightness roll
    # antialiased hard limb over ~1.5 px: opaque inside, 0 outside
    edge = 1.5 * (x[1] - x[0])
    r = np.sqrt(np.maximum(r2, 1e-12))
    img[..., 3] = np.clip((1.0 - r) / edge, 0.0, 1.0)
    return img


def sun_sprite_rgba(T, logL, npix=385, disk_frac=0.36, spike=1.0,
                    r_abs=None):
    """Disk + luminosity bloom + diffraction spikes in one RGBA sprite.

    The photospheric limb sits at r = disk_frac of the sprite half-width, so
    callers place the sprite with extent = R_disk/disk_frac to keep the limb
    geometrically exact. Bloom halo and 4-point spikes live OUTSIDE the limb
    (reads as an optical artifact, i.e. brightness, not size); both scale
    with log L.
    """
    lum = np.clip((logL + 1.0) / 4.5, 0.0, 1.0)          # 0..1 over track
    col = np.array(bb_rgb(T))
    flare_col = 0.45 * col + 0.55 * np.array([1.0, 1.0, 1.0])

    x = np.linspace(-1, 1, npix)
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx * xx + yy * yy)

    # --- bloom halo outside the limb: two-scale smooth glow, no structure ---
    d = np.maximum(r / disk_frac - 1.0, 0.0)             # 0 at limb
    s_in, s_out = 0.10 + 0.10 * lum, 0.45 + 0.70 * lum
    a_in, a_out = 0.55 + 0.40 * lum, 0.08 + 0.22 * lum
    if r_abs is not None:
        # to-scale panel: cap the halo's ABSOLUTE extent (in AU) so a
        # resolved giant carries only a thin rim glow, never false size.
        # d is in units of the disk radius, so divide AU budgets by r_abs.
        s_in = min(s_in, (0.03 + 0.03 * lum) / r_abs)
        s_out = min(s_out, (0.08 + 0.15 * lum) / r_abs)
        att = float(np.clip(0.03 / r_abs, 0.15, 1.0))
        a_in, a_out = a_in * att, a_out * att
    inner = np.exp(-d / s_in) * a_in
    outer = np.exp(-d / s_out) * a_out
    bloom = inner + outer
    # NOTE: bloom is NOT zeroed under the disk -- the opaque disk covers it,
    # and it must underfill the antialiased limb (no dark ring).
    # radial window -> exactly zero well inside the square sprite edge
    window = np.clip((0.97 - r) / 0.35, 0.0, 1.0) ** 2
    glow_a = np.clip(bloom * window, 0.0, 1.0)

    img = np.zeros((npix, npix, 4))
    img[..., :3] = flare_col
    img[..., 3] = glow_a * 0.9

    # --- composite the hard-limb disk in the centre ---
    # diameter in px = 2 * disk_frac * half-width; rounded up to odd
    n_disk = 2 * round(disk_frac * (npix - 1) / 2) + 1
    disk = sun_disk_rgba(T, n_disk)
    c0 = (npix - n_disk) // 2
    sl = slice(c0, c0 + n_disk)
    da = disk[..., 3:4]
    img[sl, sl, :3] = disk[..., :3] * da + img[sl, sl, :3] * (1 - da)
    img[sl, sl, 3] = np.clip(disk[..., 3] + img[sl, sl, 3] * (1 - disk[..., 3]),
                             0, 1)
    return img


# ---------- blackbody color ----------

def bb_rgb(T):
    """Kelvin -> approximate display RGB (Tanner Helland fit), 1000-40000 K."""
    T = np.clip(T, 1000.0, 40000.0) / 100.0
    if T <= 66:
        r = 255.0
        g = np.clip(99.4708025861 * np.log(T) - 161.1195681661, 0, 255)
        b = 0.0 if T <= 19 else np.clip(
            138.5177312231 * np.log(T - 10) - 305.0447927307, 0, 255)
    else:
        r = np.clip(329.698727446 * ((T - 60) ** -0.1332047592), 0, 255)
        g = np.clip(288.1221695283 * ((T - 60) ** -0.0755148492), 0, 255)
        b = 255.0
    return (r / 255, g / 255, b / 255)

# ---------- figure ----------

BG = "#0b0e14"
INK = "#eef1f5"
MUTED = "#aeb7c4"        # brighter than before for PPT/Zoom legibility
GRID = "#232936"

# ----- milestone captions (absolute stellar age in Gyr) -----
# one line per state; the frame shows the latest whose age <= current age.
# each caption is kept short enough to sit on one line in the left margin
MILESTONES = [
    (4.567, "Today: a 4.57-Gyr-old main-sequence star"),
    (4.68,  "+110 Myr: 1% brighter than today"),
    (5.10,  "+0.5 Gyr: CO₂ starvation — forests fail"),
    (5.34,  "+0.8 Gyr: +6% brighter; biosphere stressed"),
    (5.72,  "+1.15 Gyr: moist greenhouse — oceans boil off"),
    (6.57,  "+2 Gyr: oceans gone; surface sterilized"),
    (8.24,  "+3.7 Gyr: runaway greenhouse — a second Venus"),
    (8.995, "+4.4 Gyr: core hydrogen exhausted"),
    (9.07,  "+4.5 Gyr: Milky Way–Andromeda merger"),
    (9.25,  "Subgiant: the inert helium core contracts"),
    (9.55,  "Red-giant branch: a hydrogen shell ignites"),
    (11.21, "Red giant: twice as bright, far larger"),
    (11.85, "Upper RGB: degenerate He core, deep convection"),
    (11.9465, "Mercury is swallowed by the Sun"),
    (11.9513, "Venus is swallowed"),
    (11.9520, "Helium flash at the RGB tip (R = 0.94 AU)"),
    (11.97, "Horizontal branch: helium-core burning"),
    (12.00, "AGB: helium & hydrogen shells; second ascent"),
    (12.070, "Thermal pulses eject the envelope"),
    (12.075, "Resonant chaos destabilizes Earth & Mars"),
    (12.085, "Mars is ejected; Earth left on an eccentric orbit"),
    (12.088, "Planetary nebula lights up the ejected gas"),
    (12.093, "A white dwarf is born: 0.53 M☉, Earth-sized"),
    (12.40, "White-dwarf cooling: Earth orbits a dead ember"),
]


def milestone_for(age_yr):
    age_gyr = age_yr / 1e9
    txt = MILESTONES[0][1]
    for a, t in MILESTONES:
        if age_gyr >= a:
            txt = t
        else:
            break
    return txt


def main():
    d = resample_arclength(*load())
    n = len(d["age"])
    Teff = 10 ** d["lT"]
    R_au = 10 ** d["lR"] * RSUN_AU

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(BG)
    axL = fig.add_axes([0.055, 0.09, 0.40, 0.82])
    axR = fig.add_axes([0.50, 0.05, 0.48, 0.90])

    # --- left: HR ---
    axL.set_facecolor(BG)
    axL.plot(d["lT"], d["lL"], color=GRID, lw=1.2, zorder=1)  # future, dim
    trail, = axL.plot([], [], color="#9aa4b2", lw=1.6, zorder=2)
    axL.invert_xaxis()
    # cardinal tick labels on log-scaled axes
    T_ticks = [100000, 30000, 10000, 6000, 4000, 3000]
    axL.set_xticks(np.log10(T_ticks))
    axL.set_xticklabels([f"{t:,}" for t in T_ticks])
    L_ticks = [0.1, 1, 10, 100, 1000]
    axL.set_yticks(np.log10(L_ticks))
    axL.set_yticklabels(["0.1", "1", "10", "100", "1,000"])
    axL.set_xlabel(r"temperature  [K]", color=INK, fontsize=13)
    axL.set_ylabel(r"luminosity  [$L_\odot$]", color=INK, fontsize=13)
    for sp in axL.spines.values():
        sp.set_color(GRID)
    axL.tick_params(colors=MUTED)
    axL.grid(color=GRID, lw=0.5, alpha=0.6)
    # HR sun marker: fixed-size dot, color carries Teff; size stays constant
    sun_pt = axL.scatter([], [], s=130, zorder=5, edgecolors="#0b0e14",
                         linewidths=1.0)

    # milestone caption + readouts, centered where the track never goes
    # (the middle of the HR diagram is empty). Compact spacing, heavier weight.
    milestone_txt = axL.text(0.05, 0.400, "", transform=axL.transAxes,
                             color=INK, fontsize=13, va="top", ha="left",
                             weight="bold")
    age_txt = axL.text(0.05, 0.300, "", transform=axL.transAxes, color=MUTED,
                       fontsize=12.5, va="top", family="monospace",
                       weight="semibold")
    dt_txt = axL.text(0.05, 0.258, "", transform=axL.transAxes, color=MUTED,
                      fontsize=12.5, va="top", family="monospace",
                      weight="semibold")
    stat_txt = axL.text(0.05, 0.216, "", transform=axL.transAxes, color=MUTED,
                        fontsize=12.5, va="top", family="monospace",
                        weight="semibold")

    # --- right: solar system to scale ---
    LIM = 1.75
    axR.set_facecolor(BG)
    axR.set_aspect("equal")
    axR.set_xlim(-LIM, LIM); axR.set_ylim(-LIM, LIM)
    axR.axis("off")
    colors = {n: c for n, a, c in ORBITS}
    arch = None
    orbit_dots, planet_dots, peri_segs, name_txts = {}, {}, {}, {}
    if ARCHIVE and all(os.path.exists(p.strip())
                       for p in ARCHIVE.split(";")):
        from orbit_cloud import ArchiveOrbits, orbit_cloud_xy, periastron_seg_xy
        arch = ArchiveOrbits(ARCHIVE, ARCHIVE_T0)
        # static reference circles at the present-day semi-major axes
        theta = np.linspace(0, 2 * np.pi, 361)
        for _pn, pa, _pc in ORBITS:
            axR.plot(pa * np.cos(theta), pa * np.sin(theta), color=GRID,
                     lw=0.9, zorder=2)
        for pn, _pa, pc in ORBITS:
            orbit_dots[pn] = axR.scatter([], [], s=5.0, color=pc, alpha=0.9,
                                         linewidths=0, zorder=5)
            planet_dots[pn] = axR.scatter([], [], s=30, color=pc, zorder=7,
                                          edgecolors=BG, linewidths=1.2)
            peri_segs[pn], = axR.plot([], [], color=pc, lw=1.0, alpha=0.0,
                                      solid_capstyle="round", zorder=4)
            name_txts[pn] = axR.text(0, 0, pn, color=MUTED, fontsize=10,
                                     va="bottom", ha="left", visible=False)
    else:
        theta = np.linspace(0, 2 * np.pi, 361)
        rng = np.random.default_rng(3)
        for name, a, c in ORBITS:
            axR.plot(a * np.cos(theta), a * np.sin(theta), color=GRID, lw=1.0)
            ang = rng.uniform(0, 2 * np.pi)
            axR.scatter([a * np.cos(ang)], [a * np.sin(ang)], s=26, color=c,
                        zorder=6, edgecolors=BG, linewidths=1.5)
            axR.annotate(name, (a * np.cos(ang), a * np.sin(ang)),
                         xytext=(6, 6), textcoords="offset points",
                         color=MUTED, fontsize=10)
    axR.text(0.02, 0.98, "inner solar system — to scale (AU)",
             transform=axR.transAxes, color=MUTED, fontsize=11, va="top")
    # AU scale ticks along x
    for r_ref in (0.5, 1.0, 1.5):
        axR.plot([r_ref, r_ref], [-0.03, 0.03], color=GRID, lw=1)
        axR.text(r_ref, -0.09, f"{r_ref:g}", color=MUTED, fontsize=9,
                 ha="center")

    sun_img = axR.imshow(sun_disk_rgba(5772.0), zorder=4,
                         extent=[-0.01, 0.01, -0.01, 0.01],
                         interpolation="bilinear")

    labels = [p[1] for p in PHASES]

    DISK_FRAC = 0.36

    def frame(i):
        T = Teff[i]
        lL_i = d["lL"][i]
        # HR trail + sprite marker
        trail.set_data(d["lT"][:i + 1], d["lL"][:i + 1])
        sun_pt.set_offsets([[d["lT"][i], lL_i]])
        sun_pt.set_color([bb_rgb(T)])
        # right panel: live osculating orbits from the N-body archive
        if arch is not None:
            from orbit_cloud import orbit_cloud_xy, periastron_seg_xy
            _, snap = arch.snapshot(d["age"][i])
            alive = set()
            for body in snap:
                n_name, o = body["name"], body["orbit"]
                if n_name not in orbit_dots:
                    continue
                if not (np.isfinite(o.a) and 0 < o.a and 0 <= o.e < 0.95):
                    # unbound/escaping body: show the dot at its true
                    # position (if on-panel) but no ellipse
                    if (np.isfinite(body["xy"]).all()
                            and max(map(abs, body["xy"])) < 1.75):
                        alive.add(n_name)
                        orbit_dots[n_name].set_offsets(np.empty((0, 2)))
                        planet_dots[n_name].set_offsets([body["xy"]])
                        peri_segs[n_name].set_alpha(0.0)
                        name_txts[n_name].set_position(
                            (body["xy"][0] + 0.05, body["xy"][1] + 0.05))
                        name_txts[n_name].set_visible(True)
                    continue
                if o.a * (1 - o.e) > 2.4:      # outside the panel
                    continue
                alive.add(n_name)
                pts = orbit_cloud_xy(o, (0, 0))
                orbit_dots[n_name].set_offsets(pts)
                planet_dots[n_name].set_offsets([body["xy"]])
                seg = periastron_seg_xy(o, (0, 0), o.a * (1 - o.e))
                if seg is not None:
                    peri_segs[n_name].set_data(seg[:, 0], seg[:, 1])
                    peri_segs[n_name].set_alpha(
                        0.5 * min(1.0, o.e / 0.2))
                name_txts[n_name].set_position(
                    (body["xy"][0] + 0.05, body["xy"][1] + 0.05))
                name_txts[n_name].set_visible(True)
            for n_name in orbit_dots:
                if n_name not in alive:
                    orbit_dots[n_name].set_offsets(np.empty((0, 2)))
                    planet_dots[n_name].set_offsets(np.empty((0, 2)))
                    peri_segs[n_name].set_alpha(0.0)
                    name_txts[n_name].set_visible(False)
        # right panel: hard limb at exactly R(t); bloom outside; spikes only
        # while the disk is nearly unresolved (point sources spike, disks don't)
        r_draw = max(R_au[i], 0.010)   # display floor when tiny
        spike_att = float(np.clip(0.05 / r_draw, 0.0, 1.0))
        r_ext = r_draw / DISK_FRAC
        sun_img.set_data(sun_sprite_rgba(T, lL_i, spike=spike_att,
                                         r_abs=r_draw, npix=641))
        sun_img.set_extent([-r_ext, r_ext, -r_ext, r_ext])
        # milestone caption (latest state at or before this age)
        if i < d["i_now"]:
            milestone_txt.set_text("Pre-main-sequence → present day")
        else:
            milestone_txt.set_text(milestone_for(d["age"][i]))
        age_txt.set_text(f"age {d['age'][i]/1e9:7.4f} Gyr")
        if 0 < i < n - 1:
            dt = d["age"][i + 1] - d["age"][i]
            dt_s = f"{dt:,.0f}" if dt >= 100 else (
                f"{dt:.1f}" if dt >= 1 else f"{dt:.3f}")
            dt_txt.set_text(f"Δt  {dt_s} yr/frame")
        L_v, R_v = 10 ** d["lL"][i], 10 ** d["lR"][i]
        L_s = f"{L_v:,.0f}" if L_v >= 10 else f"{L_v:.2f}"
        R_s = f"{R_v:,.0f}" if R_v >= 10 else f"{R_v:.2f}"
        stat_txt.set_text(
            f"L {L_s} L☉   R {R_s} R☉   Teff {T:,.0f} K")
        return []

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    out = os.path.join(BASE, "sun_future_v1.mp4")
    anim.save(out, writer=FFMpegWriter(fps=30, bitrate=6000))
    print("wrote", out)


if __name__ == "__main__":
    main()
