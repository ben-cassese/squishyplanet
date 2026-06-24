"""High-precision (mpmath) regression tests for the fast transit components.

These pin the three pieces wired into
``squishyplanet.engine.polynomial_limb_darkened_transit`` against ~50-digit mpmath
references (independent ground truth, not a comparison against the previous
implementation):

- ``star_solution_vec``      -- closed-form star-arc integrals (s0, s1).
- ``planet_solution_vec``    -- fused Green's-theorem integrals (s0, s1, s2).
- ``outline_prelude``        -- direct orbital-elements -> projected-ellipse outline.

The references are ported from the prototype verification scripts in ``zzz/``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import mpmath as mp
import numpy as np

from squishyplanet.engine.polynomial_limb_darkened_transit import (
    outline_prelude,
    planet_solution_vec,
    star_solution_vec,
)

mp.mp.dps = 50


# ---------------------------------------------------------------------------
# star_solution_vec vs mpmath star-arc integrals
# ---------------------------------------------------------------------------
def _thetas(a, b, c):
    """Replicate the function's (a, b) -> (theta1, theta2) conversion in numpy."""
    cx1, cx2, cx3, cy1, cy2, cy3 = c
    x1 = cx1 * np.cos(a) + cx2 * np.sin(a) + cx3
    y1 = cy1 * np.cos(a) + cy2 * np.sin(a) + cy3
    t1 = np.arctan2(y1, x1)
    t1 = t1 + 2 * np.pi if t1 < 0 else t1
    x2 = cx1 * np.cos(b) + cx2 * np.sin(b) + cx3
    y2 = cy1 * np.cos(b) + cy2 * np.sin(b) + cy3
    t2 = np.arctan2(y2, x2)
    t2 = t2 + 2 * np.pi if t2 < 0 else t2
    return (t1, t2) if t1 < t2 else (t2, t1)


def _mp_s0(t):
    return mp.cos(t) ** 2


def _mp_s1(t):
    if (t < mp.pi / 2) or (t > 3 * mp.pi / 2):
        return (mp.pi * mp.cos(t) * (5 + 3 * mp.cos(2 * t))) / 24
    return -(mp.pi * mp.cos(t) * (1 + 3 * mp.cos(2 * t))) / 24


def _mp_int(f, lo, hi):
    """Integrate ``f`` over ``[lo, hi]``, splitting at the s1 breakpoints."""
    pts = [mp.mpf(lo)]
    for bp in (mp.pi / 2, 3 * mp.pi / 2):
        if lo < bp < hi:
            pts.append(bp)
    pts.append(mp.mpf(hi))
    return sum(mp.quad(f, [pts[i], pts[i + 1]]) for i in range(len(pts) - 1))


def _mp_star_reference(a, b, c):
    """High-precision s0, s1 for the given arc, matching star_solution_vec's logic."""
    t1, t2 = _thetas(a, b, c)
    if (t2 - t1) < np.pi:
        s0 = _mp_int(_mp_s0, t1, t2)
        s1 = _mp_int(_mp_s1, t1, t2)
    else:
        s0 = _mp_int(_mp_s0, t2, 2 * mp.pi) + _mp_int(_mp_s0, 0.0, t1)
        s1 = _mp_int(_mp_s1, t2, 2 * mp.pi) + _mp_int(_mp_s1, 0.0, t1)
    return float(s0), float(s1)


def test_star_solution_vec_matches_mpmath():
    """Closed-form star_solution_vec equals the ~50-digit reference (s0, s1).

    Cases are chosen to exercise the no-wrap and wrap branches and arcs crossing the
    pi/2 and 3*pi/2 breakpoints of the s1 integrand.
    """
    rng = np.random.default_rng(0)
    g = jnp.zeros(3)
    worst = 0.0
    for _ in range(40):
        a = float(rng.uniform(0, 2 * np.pi))
        b = float(rng.uniform(0, 2 * np.pi))
        # near-limb outline (partial-transit-like), spanning the breakpoints
        d = float(rng.uniform(0.85, 1.15))
        ang = float(rng.uniform(0, 2 * np.pi))
        r1, r2 = float(rng.uniform(0.05, 0.3)), float(rng.uniform(0.05, 0.3))
        phi = float(rng.uniform(0, np.pi))
        c = [
            r1 * np.cos(phi),
            -r2 * np.sin(phi),
            d * np.cos(ang),
            r1 * np.sin(phi),
            r2 * np.cos(phi),
            d * np.sin(ang),
        ]
        got = np.asarray(star_solution_vec(a, b, g, *c))[:2]
        ref = np.array(_mp_star_reference(a, b, c))
        worst = max(worst, float(np.max(np.abs(got - ref))))
    assert worst < 1e-9, f"star_solution_vec vs mpmath worst = {worst:.2e}"


# ---------------------------------------------------------------------------
# planet_solution_vec vs mpmath Green's-theorem integrals (s0, s1, s2)
# ---------------------------------------------------------------------------
def _mp_planet_reference(para):
    """High-precision (s0, s1, s2) for a fully-inside outline (quadratic LD)."""
    cx1, cx2, cx3 = (mp.mpf(para[k]) for k in ("c_x1", "c_x2", "c_x3"))
    cy1, cy2, cy3 = (mp.mpf(para[k]) for k in ("c_y1", "c_y2", "c_y3"))

    def x_of(s):
        return mp.cos(s) * cx1 + mp.sin(s) * cx2 + cx3

    def y_of(s):
        return mp.cos(s) * cy1 + mp.sin(s) * cy2 + cy3

    def dyds(s):
        return -mp.sin(s) * cy1 + mp.cos(s) * cy2

    def i0(s):
        return x_of(s) * dyds(s)

    def i1(s):
        x, y = x_of(s), y_of(s)
        root = mp.sqrt(1 - x**2 - y**2)
        return (
            dyds(s) * (mp.pi + 6 * x * root - 6 * mp.atan(x / root) * (-1 + y**2)) / 12
        )

    def i2(s):
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


def test_planet_solution_vec_matches_mpmath():
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
def _Rz(t):
    return mp.matrix([[mp.cos(t), -mp.sin(t), 0], [mp.sin(t), mp.cos(t), 0], [0, 0, 1]])


def _Rx(t):
    return mp.matrix([[1, 0, 0], [0, mp.cos(t), -mp.sin(t)], [0, mp.sin(t), mp.cos(t)]])


def _Ry(t):
    return mp.matrix([[mp.cos(t), 0, mp.sin(t)], [0, 1, 0], [-mp.sin(t), 0, mp.cos(t)]])


def _mp_outline_point(p, alpha):
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


def test_outline_prelude_matches_mpmath():
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
