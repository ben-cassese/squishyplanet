"""Monte-Carlo validation of the generic ellipse-star arc integrator.

The engine no longer assumes an ellipse crosses the stellar limb exactly twice:
:func:`ellipse_star_term` splits both the ellipse outline and the stellar limb into
arcs at every crossing and keeps the arcs whose midpoints lie inside the other region.
These tests validate that machinery against brute-force Monte-Carlo area estimates for
a uniformly-bright star, concentrating on the previously-broken geometries: long, thin
ellipses that "skewer" the star and cross its limb four times (the projected shape of
a nearly edge-on ring, but also reachable with the projected-ellipse
parameterization), arcs that span the parameterization seam at alpha = 0 = 2 pi, and
containment in both directions.
"""

import platform

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from squishyplanet.engine.greens_basis_transform import generate_change_of_basis_matrix
from squishyplanet.engine.parametric_ellipse import point_in_ellipse
from squishyplanet.engine.polynomial_limb_darkened_transit import (
    _single_intersection_points,
    ellipse_star_term,
    parameterize_2d_helper,
)

N_MC = 10_000_000
# 5-sigma on the binomial area estimate: pi * 5 * 0.5 / sqrt(N)
MC_TOL = jnp.pi * 5 * 0.5 / jnp.sqrt(N_MC)

# uniform limb darkening: u = (0, 0)
_G_COEFFS = jnp.matmul(generate_change_of_basis_matrix(2), jnp.array([-1.0, 0.0, 0.0]))
_NORMALIZATION = 1 / (jnp.pi * (_G_COEFFS[0] + (2 / 3) * _G_COEFFS[1]))

# fixed sample of the unit disk, shared across all cases
_KEY = jax.random.PRNGKey(0)
_K1, _K2 = jax.random.split(_KEY)
_MC_R = jnp.sqrt(jax.random.uniform(_K1, (N_MC,)))
_MC_ANG = jax.random.uniform(_K2, (N_MC,), maxval=2 * jnp.pi)
_MC_X = _MC_R * jnp.cos(_MC_ANG)
_MC_Y = _MC_R * jnp.sin(_MC_ANG)


def _ellipse(r1: float, r2: float, theta: float, xc: float, yc: float) -> tuple:
    """Implicit and parametric coefficient dicts for one projected ellipse."""
    two, para = parameterize_2d_helper(
        jnp.array(r1),
        jnp.array(1 - r2 / r1),
        jnp.array(theta),
        jnp.array(xc),
        jnp.array(yc),
    )
    return two, para


def _blocked_area(para: dict, two: dict) -> float:
    """Area of (ellipse intersect star) from the Green's-theorem machinery."""
    blocked_flux = ellipse_star_term(para, two, _G_COEFFS) * _NORMALIZATION
    return float(blocked_flux * jnp.pi)


@jax.jit
def _mc_area(para: dict) -> jax.Array:
    """Monte-Carlo area of (ellipse intersect star) from disk samples."""
    inside = point_in_ellipse(_MC_X, _MC_Y, **para)
    return jnp.pi * jnp.mean(inside)


def _n_crossings(two: dict) -> int:
    xs, _ = _single_intersection_points(**two)
    return int(jnp.sum(xs != 999))


def test_thin_ellipses_match_monte_carlo() -> None:
    """Random thin "skewer" ellipses agree with MC areas, including 4-crossing cases.

    This is a regression test for the old 2-intersection assumption, which returned
    incorrect fluxes whenever a long thin projected ellipse crossed the limb four
    times.
    """
    rng = np.random.default_rng(1)
    n_four = 0
    worst = 0.0
    for _ in range(60):
        r1 = float(rng.uniform(0.8, 4.0))
        r2 = float(rng.uniform(0.005, 0.3))
        theta = float(rng.uniform(0, np.pi))
        d = float(rng.uniform(0.0, 1.2))
        ang = float(rng.uniform(0, 2 * np.pi))
        two, para = _ellipse(r1, r2, theta, d * np.cos(ang), d * np.sin(ang))
        if _n_crossings(two) == 4:
            n_four += 1
        err = abs(_blocked_area(para, two) - float(_mc_area(para)))
        worst = max(worst, err)
    assert n_four >= 10, f"random draw produced too few 4-crossing cases: {n_four}"
    assert worst < MC_TOL, f"worst |area - MC| = {worst:.2e} (tol {MC_TOL:.2e})"


def test_ordinary_ellipses_match_monte_carlo() -> None:
    """Planet-like (0- and 2-crossing) ellipses also agree with MC areas."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(40):
        r1 = float(rng.uniform(0.02, 0.4))
        r2 = float(rng.uniform(0.3, 1.0)) * r1
        theta = float(rng.uniform(0, np.pi))
        d = float(rng.uniform(0.0, 1.0 + r1))
        ang = float(rng.uniform(0, 2 * np.pi))
        two, para = _ellipse(r1, r2, theta, d * np.cos(ang), d * np.sin(ang))
        err = abs(_blocked_area(para, two) - float(_mc_area(para)))
        worst = max(worst, err)
    assert worst < MC_TOL, f"worst |area - MC| = {worst:.2e} (tol {MC_TOL:.2e})"


def _shift_parameter_phase(para: dict, phase: float) -> dict:
    """Reparameterize the same ellipse with ``alpha -> alpha + phase``.

    Preserves the curve and its counterclockwise orientation but moves the
    ``alpha = 0 = 2 pi`` seam to what was previously ``alpha = phase``.
    """
    c, s = jnp.cos(phase), jnp.sin(phase)
    return {
        "c_x1": para["c_x1"] * c + para["c_x2"] * s,
        "c_x2": -para["c_x1"] * s + para["c_x2"] * c,
        "c_x3": para["c_x3"],
        "c_y1": para["c_y1"] * c + para["c_y2"] * s,
        "c_y2": -para["c_y1"] * s + para["c_y2"] * c,
        "c_y3": para["c_y3"],
    }


def test_wrap_arc_regression() -> None:
    """4-crossing geometries whose inside-star arc spans the alpha = 0 seam.

    The parametric angle alpha = 0 falls at the point (c_x1 + c_x3, c_y1 + c_y3),
    which :func:`poly_to_parametric` places at the major-axis tip -- outside the star
    for a skewer. Shifting the parameter phase by pi/2 moves the seam to the
    minor-axis tip, which for a near-centered skewer is inside the star, forcing an
    inside-star arc to span the seam. The abandoned implementation on the 20-rings
    branch paired the sorted crossing angles cyclically with ``jnp.roll`` and
    mis-handled exactly this wrap arc (antipodal test midpoint, reversed integration
    limits); the bookend scheme splits it at the seam instead.
    """
    rng = np.random.default_rng(3)
    n_hit = 0
    worst = 0.0
    for _ in range(40):
        r1 = float(rng.uniform(1.5, 3.0))
        r2 = float(rng.uniform(0.05, 0.25))
        theta = float(rng.uniform(0, np.pi))
        # small center offset so the major axis pierces the star
        d = float(rng.uniform(0.0, 0.4))
        ang = float(rng.uniform(0, 2 * np.pi))
        two, para = _ellipse(r1, r2, theta, d * np.cos(ang), d * np.sin(ang))
        para = _shift_parameter_phase(para, np.pi / 2)
        seam_inside = (
            float(para["c_x1"] + para["c_x3"]) ** 2
            + float(para["c_y1"] + para["c_y3"]) ** 2
            < 1.0
        )
        if not (seam_inside and _n_crossings(two) == 4):
            continue
        n_hit += 1
        err = abs(_blocked_area(para, two) - float(_mc_area(para)))
        worst = max(worst, err)
    assert n_hit >= 5, f"too few seam-inside 4-crossing draws: {n_hit}"
    assert worst < MC_TOL, f"worst |area - MC| = {worst:.2e} (tol {MC_TOL:.2e})"


def test_containment_both_directions() -> None:
    """Ellipse fully inside the star, and ellipse fully containing the star."""
    # small ellipse fully inside: area = pi r1 r2
    two, para = _ellipse(0.2, 0.1, 0.7, 0.1, -0.2)
    assert abs(_blocked_area(para, two) - np.pi * 0.2 * 0.1) < 1e-12

    # huge ellipse containing the whole star: area = pi (the star is covered)
    two, para = _ellipse(3.0, 2.0, 0.3, 0.2, 0.1)
    assert abs(_blocked_area(para, two) - np.pi) < 1e-12

    # fully outside: zero
    two, para = _ellipse(0.2, 0.1, 0.0, 2.0, 0.0)
    assert _blocked_area(para, two) == 0.0


@pytest.mark.skipif(
    platform.machine() not in ("arm64", "aarch64"),
    reason=(
        "jnp.roots' eig-based quartic solve is ill-conditioned near tangency on x86 "
        "LAPACK (same root cause as test_pathological_transits.py's "
        "test_exact_tangency_negative_blocked_flux_known_issue) and segfaults the "
        "interpreter partway through this dense sweep on x86 runners; clean on arm64."
    ),
)
def test_continuity_through_crossing_transitions() -> None:
    """No spikes as a thin ellipse slides through 0 <-> 2 <-> 4 crossing transitions.

    Misclassified arcs would produce O(0.1-1) jumps in the blocked area; smooth motion
    of this geometry changes the area by at most ~perimeter * step. Sweep the center
    through the full approach with a dense grid and bound adjacent differences.
    """
    ds = np.linspace(0.0, 2.6, 2000)
    areas = []
    for d in ds:
        two, para = _ellipse(
            1.5, 0.02, 0.4, d * np.cos(0.4 + np.pi / 2), d * np.sin(0.4 + np.pi / 2)
        )
        areas.append(_blocked_area(para, two))
    areas = np.array(areas)
    assert np.all(np.isfinite(areas))
    assert np.all(areas >= -1e-12)
    assert np.all(areas <= np.pi + 1e-12)
    steps = np.abs(np.diff(areas))
    assert steps.max() < 2e-2, f"area jump {steps.max():.3e} at transition"
