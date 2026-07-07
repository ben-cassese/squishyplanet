"""Unit tests for the ring geometry primitives.

Covers the parametric/implicit conversions in ``parametric_ellipse``, the
Mathematica-derived ring-edge projection in ``rings.ring_para_coeffs`` (including its
counterclockwise-orientation guarantee, which the Green's theorem integrals rely on),
and the general parametric-vs-implicit conic intersection solver.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from squishyplanet.engine.kepler import skypos
from squishyplanet.engine.parametric_ellipse import (
    parametric_to_poly,
    point_in_ellipse,
    poly_to_parametric,
)
from squishyplanet.engine.planet_2d import planet_2d_coeffs
from squishyplanet.engine.planet_3d import planet_3d_coeffs
from squishyplanet.engine.ringed_transit import STAR_TWO
from squishyplanet.engine.rings import (
    para_eval,
    parametric_conic_intersections,
    ring_para_coeffs,
)


def _random_transiting_planet(rng: np.random.Generator) -> tuple[dict, dict, dict]:
    """A random planet whose projected center lies within 2 Rstar of the star."""
    while True:
        p = {
            "a": jnp.array([rng.uniform(2.0, 20.0)]),
            "e": jnp.array([rng.uniform(0.0, 0.7)]),
            "f": jnp.array([rng.uniform(0, 2 * np.pi)]),
            "Omega": jnp.array([np.pi]),
            "i": jnp.array([np.pi / 2 + rng.uniform(-0.1, 0.1)]),
            "omega": jnp.array([rng.uniform(0, 2 * np.pi)]),
            "obliq": jnp.array([rng.uniform(-np.pi / 2, np.pi / 2)]),
            "prec": jnp.array([rng.uniform(0, 2 * np.pi)]),
            "r": jnp.array([rng.uniform(0.01, 0.3)]),
        }
        pos = skypos(**p)
        if float(pos[0, 0] ** 2 + pos[1, 0] ** 2) < 4.0:
            break
    f1 = jnp.array([rng.uniform(0.0, 0.5)])
    f2 = jnp.array([rng.uniform(0.0, 0.5)]) * (1 - f1)
    three = planet_3d_coeffs(**p, f1=f1, f2=f2)
    two = planet_2d_coeffs(**three)
    para = poly_to_parametric(**two)
    return p, two, para


def _random_ring(rng: np.random.Generator, p: dict, r_lo: float = 0.05) -> dict:
    return ring_para_coeffs(
        a=p["a"],
        e=p["e"],
        f=p["f"],
        Omega=p["Omega"],
        i=p["i"],
        omega=p["omega"],
        rRing=jnp.array([rng.uniform(r_lo, 2.0)]),
        ring_obliq=jnp.array([rng.uniform(-np.pi, np.pi)]),
        ring_prec=jnp.array([rng.uniform(0, 2 * np.pi)]),
    )


def test_parametric_to_poly_round_trip() -> None:
    """poly -> para -> poly round-trips, and the returned implicit form is satisfied
    on points sampled from the parametric curve.

    Restricted to transit-relevant geometry (projected center within 2 Rstar): the
    "= 1" implicit convention inherently loses ~(d/r)^2 * eps precision when the
    ellipse sits far from the origin, which the transit path never sees.
    """
    rng = np.random.default_rng(0)
    alphas = jnp.linspace(0, 2 * np.pi, 100)
    worst_rt = 0.0
    worst_direct = 0.0
    for _ in range(100):
        _, two, para = _random_transiting_planet(rng)
        two_back = parametric_to_poly(**para)
        for name in two:
            rel = jnp.max(
                jnp.abs(two[name] - two_back[name]) / (jnp.abs(two[name]) + 1e-15)
            )
            worst_rt = max(worst_rt, float(rel))
        x, y = para_eval(alphas, {k: v[0] for k, v in para.items()})
        val = (
            two_back["rho_xx"][0] * x**2
            + two_back["rho_xy"][0] * x * y
            + two_back["rho_x0"][0] * x
            + two_back["rho_yy"][0] * y**2
            + two_back["rho_y0"][0] * y
            + two_back["rho_00"][0]
        )
        worst_direct = max(worst_direct, float(jnp.max(jnp.abs(val - 1))))
    assert worst_rt < 1e-8, f"round trip rel err {worst_rt:.2e}"
    assert worst_direct < 1e-10, f"direct implicit residual {worst_direct:.2e}"


def test_ring_para_coeffs_orientation_center_axis() -> None:
    """Ring outlines are CCW (positive det), centered on the planet, with projected
    semi-major axis exactly rRing."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        p, _, _ = _random_transiting_planet(rng)
        rRing = jnp.array([rng.uniform(0.05, 2.0)])
        coeffs = ring_para_coeffs(
            a=p["a"],
            e=p["e"],
            f=p["f"],
            Omega=p["Omega"],
            i=p["i"],
            omega=p["omega"],
            rRing=rRing,
            ring_obliq=jnp.array([rng.uniform(-np.pi, np.pi)]),
            ring_prec=jnp.array([rng.uniform(0, 2 * np.pi)]),
        )
        det = coeffs["c_x1"] * coeffs["c_y2"] - coeffs["c_x2"] * coeffs["c_y1"]
        assert float(det[0]) >= 0, "ring outline is clockwise"
        pos = skypos(**p)
        assert abs(float(coeffs["c_x3"][0] - pos[0, 0])) < 1e-12
        assert abs(float(coeffs["c_y3"][0] - pos[1, 0])) < 1e-12
        M = jnp.array(
            [
                [coeffs["c_x1"][0], coeffs["c_x2"][0]],
                [coeffs["c_y1"][0], coeffs["c_y2"][0]],
            ]
        )
        smax = jnp.linalg.svd(M, compute_uv=False)[0]
        assert abs(float(smax - rRing[0]) / float(rRing[0])) < 1e-10


def test_ring_shoelace_area_positive() -> None:
    """The signed (shoelace) area of a sampled ring outline is +pi * r1 * r2."""
    rng = np.random.default_rng(2)
    alphas = jnp.linspace(0, 2 * np.pi, 20001)
    for _ in range(20):
        p, _, _ = _random_transiting_planet(rng)
        coeffs = _random_ring(rng, p)
        flat = {k: v[0] for k, v in coeffs.items()}
        x, y = para_eval(alphas, flat)
        shoelace = 0.5 * jnp.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        det = flat["c_x1"] * flat["c_y2"] - flat["c_x2"] * flat["c_y1"]
        expected = jnp.pi * det
        assert float(jnp.abs(shoelace - expected)) < 1e-5 * max(
            1.0, abs(float(expected))
        )


def test_point_in_ellipse_matches_implicit() -> None:
    """The adjugate membership test agrees with the implicit-conic evaluation for
    well-conditioned ellipses, and a degenerate ellipse has an empty interior."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        p, two, para = _random_transiting_planet(rng)
        px = para["c_x3"] + rng.normal(size=200) * float(p["r"][0]) * 2
        py = para["c_y3"] + rng.normal(size=200) * float(p["r"][0]) * 2
        inside_para = point_in_ellipse(px, py, **{k: v[0] for k, v in para.items()})
        val = (
            two["rho_xx"] * px**2
            + two["rho_xy"] * px * py
            + two["rho_x0"] * px
            + two["rho_yy"] * py**2
            + two["rho_y0"] * py
            + two["rho_00"]
        )
        assert bool(jnp.all(inside_para == (val < 1)))

    degenerate = {
        "c_x1": 1.0,
        "c_x2": 0.0,
        "c_x3": 0.0,
        "c_y1": 0.5,
        "c_y2": 0.0,
        "c_y3": 0.0,
    }
    res = point_in_ellipse(jnp.array([0.0, 0.5]), jnp.array([0.0, 0.25]), **degenerate)
    assert not bool(jnp.any(res))


def test_parametric_conic_intersections_residuals() -> None:
    """Intersection points satisfy both curve equations, and the returned angles map
    back onto the same points, including for nearly-degenerate rings."""
    rng = np.random.default_rng(4)
    n_found = 0
    for _ in range(150):
        p, two, _ = _random_transiting_planet(rng)
        ring = _random_ring(rng, p, r_lo=float(p["r"][0]) * 1.05)
        ring_flat = {k: v[0] for k, v in ring.items()}
        for conic in ({k: v[0] for k, v in two.items()}, STAR_TWO):
            alphas, xs, ys = parametric_conic_intersections(**ring_flat, **conic)
            good = xs != 999
            if not bool(jnp.any(good)):
                continue
            n_found += int(jnp.sum(good))
            val = (
                conic["rho_xx"] * xs**2
                + conic["rho_xy"] * xs * ys
                + conic["rho_x0"] * xs
                + conic["rho_yy"] * ys**2
                + conic["rho_y0"] * ys
                + conic["rho_00"]
            )
            assert float(jnp.max(jnp.where(good, jnp.abs(val - 1), 0.0))) < 1e-6
            rx, ry = para_eval(alphas, ring_flat)
            resid = jnp.sqrt((rx - xs) ** 2 + (ry - ys) ** 2)
            assert float(jnp.max(jnp.where(good, resid, 0.0))) < 1e-6
    assert n_found > 50, f"suspiciously few intersections: {n_found}"

    # nearly edge-on but open ring (thin sliver) against a nearby planet
    p = {
        "a": jnp.array([6.0]),
        "e": jnp.array([0.0]),
        "f": jnp.array([np.pi / 2]),
        "Omega": jnp.array([np.pi]),
        "i": jnp.array([np.pi / 2]),
        "omega": jnp.array([0.0]),
        "obliq": jnp.array([0.0]),
        "prec": jnp.array([0.0]),
        "r": jnp.array([0.1]),
    }
    three = planet_3d_coeffs(**p, f1=jnp.array([0.0]), f2=jnp.array([0.0]))
    two = {k: v[0] for k, v in planet_2d_coeffs(**three).items()}
    for eps in (1e-3, 1e-6, 1e-9):
        ring = ring_para_coeffs(
            a=p["a"],
            e=p["e"],
            f=p["f"],
            Omega=p["Omega"],
            i=p["i"],
            omega=p["omega"],
            rRing=jnp.array([0.2]),
            ring_obliq=jnp.array([eps]),
            ring_prec=jnp.array([np.pi / 2]),
        )
        ring_flat = {k: v[0] for k, v in ring.items()}
        _, xs, ys = parametric_conic_intersections(**ring_flat, **two)
        good = xs != 999
        assert int(jnp.sum(good)) == 4, f"expected 4 crossings at eps={eps}"
        val = (
            two["rho_xx"] * xs**2
            + two["rho_xy"] * xs * ys
            + two["rho_x0"] * xs
            + two["rho_yy"] * ys**2
            + two["rho_y0"] * ys
            + two["rho_00"]
        )
        assert float(jnp.max(jnp.where(good, jnp.abs(val - 1), 0.0))) < 1e-6
