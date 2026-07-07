"""High-precision (mpmath) regression tests for the fast transit components.

These pin the three pieces wired into
``squishyplanet.engine.polynomial_limb_darkened_transit`` against ~50-digit mpmath
references (independent ground truth, not a comparison against the previous
implementation):

- ``star_arc_solution_vec``  -- closed-form star-arc integrals (s0, s1).
- ``planet_solution_vec``    -- fused Green's-theorem integrals (s0, s1, s2).
- ``outline_prelude``        -- direct orbital-elements -> projected-ellipse outline.

The references are ported from the prototype verification scripts in ``zzz/``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import mpmath as mp
import numpy as np

import squishyplanet.engine.polynomial_limb_darkened_transit as pld
from squishyplanet import OblateSystem
from squishyplanet.engine.kepler import skypos
from squishyplanet.engine.polynomial_limb_darkened_transit import (
    lightcurve,
    outline_prelude,
    planet_solution_vec,
    star_arc_solution_vec,
)

mp.mp.dps = 50


# ---------------------------------------------------------------------------
# star_arc_solution_vec vs mpmath star-arc integrals
# ---------------------------------------------------------------------------
def _mp_s0(t: object) -> object:
    return mp.cos(t) ** 2


def _mp_s1(t: object) -> object:
    if (t < mp.pi / 2) or (t > 3 * mp.pi / 2):
        return (mp.pi * mp.cos(t) * (5 + 3 * mp.cos(2 * t))) / 24
    return -(mp.pi * mp.cos(t) * (1 + 3 * mp.cos(2 * t))) / 24


def _mp_int(f: object, lo: object, hi: object) -> object:
    """Integrate ``f`` over ``[lo, hi]``, splitting at the s1 breakpoints."""
    pts = [mp.mpf(lo)]
    for bp in (mp.pi / 2, 3 * mp.pi / 2):
        if lo < bp < hi:
            pts.append(bp)
    pts.append(mp.mpf(hi))
    return sum(mp.quad(f, [pts[i], pts[i + 1]]) for i in range(len(pts) - 1))


def _mp_star_reference(lo: float, hi: float) -> tuple[float, float]:
    """High-precision s0, s1 for the star arc [lo, hi] in polar angle."""
    s0 = _mp_int(_mp_s0, lo, hi)
    s1 = _mp_int(_mp_s1, lo, hi)
    return float(s0), float(s1)


def test_star_arc_solution_vec_matches_mpmath() -> None:
    """Closed-form star_arc_solution_vec equals the ~50-digit reference (s0, s1).

    Cases are chosen to exercise arcs crossing the pi/2 and 3*pi/2 breakpoints of the
    s1 integrand, plus the degenerate zero-length and full-circle arcs.
    """
    rng = np.random.default_rng(0)
    g = jnp.zeros(3)
    worst = 0.0
    lows = [0.0, 2 * np.pi]
    highs = [0.0, 2 * np.pi]
    for _ in range(40):
        t1, t2 = sorted(rng.uniform(0, 2 * np.pi, size=2))
        lows.append(float(t1))
        highs.append(float(t2))
    for lo, hi in zip(lows, highs, strict=True):
        got = np.asarray(star_arc_solution_vec(lo, hi, g))[:2]
        ref = np.array(_mp_star_reference(lo, hi))
        worst = max(worst, float(np.max(np.abs(got - ref))))
    assert worst < 1e-9, f"star_arc_solution_vec vs mpmath worst = {worst:.2e}"


# ---------------------------------------------------------------------------
# planet_solution_vec vs mpmath Green's-theorem integrals (s0, s1, s2)
# ---------------------------------------------------------------------------
def _mp_planet_reference(para: dict[str, float]) -> np.ndarray:
    """High-precision (s0, s1, s2) for a fully-inside outline (quadratic LD)."""
    cx1, cx2, cx3 = (mp.mpf(para[k]) for k in ("c_x1", "c_x2", "c_x3"))
    cy1, cy2, cy3 = (mp.mpf(para[k]) for k in ("c_y1", "c_y2", "c_y3"))

    def x_of(s: object) -> object:
        return mp.cos(s) * cx1 + mp.sin(s) * cx2 + cx3

    def y_of(s: object) -> object:
        return mp.cos(s) * cy1 + mp.sin(s) * cy2 + cy3

    def dyds(s: object) -> object:
        return -mp.sin(s) * cy1 + mp.cos(s) * cy2

    def i0(s: object) -> object:
        return x_of(s) * dyds(s)

    def i1(s: object) -> object:
        x, y = x_of(s), y_of(s)
        root = mp.sqrt(1 - x**2 - y**2)
        return (
            dyds(s) * (mp.pi + 6 * x * root - 6 * mp.atan(x / root) * (-1 + y**2)) / 12
        )

    def i2(s: object) -> object:
        common = -(
            cx3 * (mp.sin(s) * cy1 - mp.cos(s) * cy2)
            + cx2 * (cy1 + mp.cos(s) * cy3)
            - cx1 * (cy2 + mp.sin(s) * cy3)
        )
        return (1 - x_of(s) ** 2 - y_of(s) ** 2) * common

    return np.array(
        [
            float(mp.quad(i0, [0, mp.pi, 2 * mp.pi])),
            float(mp.quad(i1, [0, mp.pi / 2, mp.pi, 3 * mp.pi / 2, 2 * mp.pi])),
            float(mp.quad(i2, [0, mp.pi, 2 * mp.pi])),
        ]
    )


def test_planet_solution_vec_matches_mpmath() -> None:
    """Fused planet_solution_vec matches the ~50-digit reference over a full loop.

    Uses a near-grazing fully-inside ellipse (so the integrand's sqrt(1 - x^2 - y^2)
    nearly vanishes at one outline point) -- the hardest case for the quadrature. The
    default tolerance is 1e-8, so we allow a generous margin still far under 1 ppm.
    """
    th, r1, r2 = 0.6, 0.18, 0.11
    d = (1.0 - max(r1, r2)) * (1.0 - 1e-4)  # just inside the limb
    phi = 0.9
    para = {
        "c_x1": r1 * np.cos(th),
        "c_x2": -r2 * np.sin(th),
        "c_x3": d * np.cos(phi),
        "c_y1": r1 * np.sin(th),
        "c_y2": r2 * np.cos(th),
        "c_y3": d * np.sin(phi),
    }
    g = jnp.array([0.6, 0.3, 0.1])  # quadratic LD -> orders 0, 1, 2
    got = np.asarray(planet_solution_vec(0.0, 2 * np.pi, g, **para))
    ref = _mp_planet_reference(para)
    worst = float(np.max(np.abs(got - ref)))
    assert worst < 1e-6, f"planet_solution_vec vs mpmath worst = {worst:.2e}"


# ---------------------------------------------------------------------------
# outline_prelude vs mpmath direct outline points
# ---------------------------------------------------------------------------
def _Rz(t: object) -> object:
    return mp.matrix([[mp.cos(t), -mp.sin(t), 0], [mp.sin(t), mp.cos(t), 0], [0, 0, 1]])


def _Rx(t: object) -> object:
    return mp.matrix([[1, 0, 0], [0, mp.cos(t), -mp.sin(t)], [0, mp.sin(t), mp.cos(t)]])


def _Ry(t: object) -> object:
    return mp.matrix([[mp.cos(t), 0, mp.sin(t)], [0, 1, 0], [-mp.sin(t), 0, mp.cos(t)]])


def _mp_outline_point(p: dict[str, float], alpha: float) -> tuple[float, float]:
    """High-precision (x, y) of the outline at one alpha (same convention)."""
    R = _Rz(p["Omega"]) * _Rx(p["i"]) * _Rz(p["omega"] + p["prec"]) * _Ry(p["obliq"])
    r = mp.mpf(p["r"])
    D = mp.matrix(
        [
            [1 / r**2, 0, 0],
            [0, 1 / (r * (1 - p["f2"])) ** 2, 0],
            [0, 0, 1 / (r * (1 - p["f1"])) ** 2],
        ]
    )
    M = R * D * R.T
    axx = M[0, 0] - M[0, 2] * M[2, 0] / M[2, 2]
    ayy = M[1, 1] - M[1, 2] * M[2, 1] / M[2, 2]
    axy = M[0, 1] - M[0, 2] * M[2, 1] / M[2, 2]
    rho_xx, rho_yy, rho_xy = axx, ayy, 2 * axy
    theta = mp.mpf("0.5") * mp.atan2(rho_xy, rho_xx - rho_yy) + mp.pi / 2
    if theta < 0:
        theta += mp.pi
    cosa, sina = mp.cos(theta), mp.sin(theta)
    aa = rho_xx * cosa**2 + rho_xy * cosa * sina + rho_yy * sina**2
    bb = rho_xx * sina**2 - rho_xy * cosa * sina + rho_yy * cosa**2
    r1, r2 = 1 / mp.sqrt(aa), 1 / mp.sqrt(bb)
    ome2 = mp.mpf(p["a"]) * (-1 + mp.mpf(p["e"]) ** 2)
    cf = mp.cos(p["f"])
    denom = 1 + p["e"] * cf
    xc = (
        ome2
        * (
            mp.sin(p["f"])
            * (
                mp.cos(p["Omega"]) * mp.sin(p["omega"])
                + mp.cos(p["i"]) * mp.cos(p["omega"]) * mp.sin(p["Omega"])
            )
            + cf
            * (
                -(mp.cos(p["omega"]) * mp.cos(p["Omega"]))
                + mp.cos(p["i"]) * mp.sin(p["omega"]) * mp.sin(p["Omega"])
            )
        )
        / denom
    )
    yc = -(
        ome2
        * (
            mp.cos(p["i"]) * mp.cos(p["Omega"]) * mp.sin(p["f"] + p["omega"])
            + mp.cos(p["f"] + p["omega"]) * mp.sin(p["Omega"])
        )
        / denom
    )
    cx1, cx2 = r1 * cosa, -r2 * sina
    cy1, cy2 = r1 * sina, r2 * cosa
    x = cx1 * mp.cos(alpha) + cx2 * mp.sin(alpha) + xc
    y = cy1 * mp.cos(alpha) + cy2 * mp.sin(alpha) + yc
    return float(x), float(y)


def test_outline_prelude_matches_mpmath() -> None:
    """Direct outline_prelude reproduces the ~50-digit reference outline points."""
    rng = np.random.default_rng(7)
    tp = 2 * np.pi
    alphas = np.linspace(0, tp, 8, endpoint=False)
    worst = 0.0
    for _ in range(12):
        p = {
            "a": float(rng.uniform(5, 40)),
            "e": float(rng.uniform(0, 0.6)),
            "f": float(rng.uniform(-tp, tp)),
            "Omega": float(rng.uniform(-tp, tp)),
            "i": float(rng.uniform(0.1, np.pi - 0.1)),
            "omega": float(rng.uniform(-tp, tp)),
            "r": float(rng.uniform(0.02, 0.3)),
            "obliq": float(rng.uniform(-np.pi / 2, np.pi / 2)),
            "prec": float(rng.uniform(-tp, tp)),
            "f1": float(rng.uniform(0, 0.5)),
            "f2": float(rng.uniform(0, 0.5)),
        }
        state = {k: jnp.array([v]) for k, v in p.items()}
        state["tidally_locked"] = False
        _, para = outline_prelude(state)
        for alpha in alphas:
            x = float(
                para["c_x1"][0] * np.cos(alpha)
                + para["c_x2"][0] * np.sin(alpha)
                + para["c_x3"][0]
            )
            y = float(
                para["c_y1"][0] * np.cos(alpha)
                + para["c_y2"][0] * np.sin(alpha)
                + para["c_y3"][0]
            )
            gx, gy = _mp_outline_point(p, float(alpha))
            worst = max(worst, float(np.hypot(x - gx, y - gy)))
    assert worst < 1e-10, f"outline_prelude vs mpmath worst = {worst:.2e}"


# ---------------------------------------------------------------------------
# scheme handoff continuity (smoothstep-GL partial arc <-> trapezoid full circle)
# ---------------------------------------------------------------------------
def _gold_lightcurve(state: dict[str, object]) -> np.ndarray:
    """Recompute the light curve with both quadrature rules cranked to ~machine
    precision, giving the (physically continuous) truth to compare against.

    Temporarily swaps the module-level node tables and clears the JIT cache so the
    swap takes effect, restoring everything afterward.
    """
    saved = (pld._GL_PHI, pld._GL_WDPHI, pld._TRAP_NODES, pld._TRAP_WEIGHT)
    try:
        n, w = np.polynomial.legendre.leggauss(200)
        pld._GL_PHI = jnp.asarray((3.0 * n - n**3) / 2.0)
        pld._GL_WDPHI = jnp.asarray(w * (3.0 - 3.0 * n**2) / 2.0)
        pld._TRAP_NODES = jnp.asarray(
            np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
        )
        pld._TRAP_WEIGHT = 2.0 * np.pi / 4096
        jax.clear_caches()
        return np.asarray(lightcurve(state, False))
    finally:
        pld._GL_PHI, pld._GL_WDPHI, pld._TRAP_NODES, pld._TRAP_WEIGHT = saved
        jax.clear_caches()


def test_scheme_handoff_is_continuous() -> None:
    """No flux discontinuity where the two quadrature schemes meet.

    ``planet_solution_vec`` integrates partial-transit arcs with a smoothstep
    Gauss-Legendre rule and the full circle with a periodic trapezoid. They hand
    off at internal tangency (``d_center = 1 - r``, i.e. second/third contact),
    which is also the most grazing the full-circle integrand ever gets. A large
    planet sampled finely straddling tangency is the worst case; we check that the
    production light curve tracks the machine-precision ``_gold_lightcurve`` across
    the handoff and that the scheme-induced step is far below 1 ppm.
    """
    r = 0.25
    base = dict(
        t0=0.0,
        period=15.0,
        a=15.0,
        e=0.0,
        i=jnp.pi / 2,
        Omega=jnp.pi,
        omega=0.0,
        f1=0.0,
        f2=0.0,
        r=r,
        obliq=0.0,
        prec=0.0,
        ld_u_coeffs=jnp.array([0.5, 0.2]),
        tidally_locked=False,
    )

    # locate the internal-tangency time (d_center = 1 - r) on the ingress side
    tt = jnp.linspace(-1.0, 0.0, 400001)
    pos = np.asarray(skypos(**OblateSystem(times=tt, **base)._state))
    dc = np.hypot(pos[0], pos[1])
    front = pos[2] > 0
    tc = float(
        np.asarray(tt)[int(np.argmin(np.where(front, np.abs(dc - (1 - r)), 9.0)))]
    )

    # dense window straddling tangency
    tw = jnp.linspace(tc - 2e-4, tc + 2e-4, 4001)
    state = OblateSystem(times=tw, **base)._state
    prod = np.asarray(lightcurve(state, False))
    gold = _gold_lightcurve(state)

    # production tracks the continuous truth across the handoff
    worst = float(np.max(np.abs(prod - gold)))
    assert worst < 5e-11, f"max |prod - gold| across handoff = {worst:.2e}"

    # the scheme-induced flux step across the handoff is far below 1 ppm
    dcw = np.hypot(*np.asarray(skypos(**state))[:2])
    h = int(np.argmin(np.abs(dcw - (1 - r))))
    err = prod - gold
    step = abs(np.mean(err[h + 3 : h + 60]) - np.mean(err[h - 60 : h - 3]))
    assert step < 5e-11, f"handoff flux step = {step:.2e}"
