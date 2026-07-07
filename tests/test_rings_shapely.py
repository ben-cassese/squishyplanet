"""High-precision polygon-clipping cross-checks for the ringed-transit terms.

Complements the Monte-Carlo tests in ``test_ringed_lightcurve.py``: shapely's exact
polygon clipping on fine (4096-gon) discretizations gives blocked areas to ~1e-6
relative, roughly three orders of magnitude tighter than MC, catching sliver-level
arc-selection mistakes that MC noise would hide. Uniform limb darkening only (areas).

shapely is an optional test dependency; these tests skip if it isn't installed.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import Point, Polygon
from test_ringed_lightcurve import _make_state, _per_timestep_geometry

from squishyplanet.engine.greens_basis_transform import generate_change_of_basis_matrix
from squishyplanet.engine.polynomial_limb_darkened_transit import (
    _lightcurve_setup,
    _single_intersection_points,
)
from squishyplanet.engine.ringed_transit import (
    STAR_TWO,
    ring_star_term,
    ringed_lightcurve,
    triple_term,
)
from squishyplanet.engine.rings import (
    para_eval,
    parametric_conic_intersections,
)

N_GON = 4096
_ALPHAS = jnp.linspace(0, 2 * np.pi, N_GON, endpoint=False)
_STAR_POLY = Point(0, 0).buffer(1.0, quad_segs=N_GON // 4)
# discretization error of an inscribed N-gon vs the smooth curve, ~2pi^3/(3 N^2) for
# the unit circle and proportionally smaller for the (smaller) planet/ring curves
AREA_TOL = 5e-5


def _poly(para: dict) -> Polygon:
    x, y = para_eval(_ALPHAS, para)
    return Polygon(np.stack((np.asarray(x), np.asarray(y)), axis=1)).buffer(0)


def test_terms_and_lightcurve_match_shapely() -> None:
    """Pair terms, triple terms, and the assembled uniform-LD blocked fraction agree
    with exact polygon clipping across random ringed systems."""
    g_coeffs = jnp.matmul(
        generate_change_of_basis_matrix(2), jnp.array([-1.0, 0.0, 0.0])
    )
    norm = 1 / (jnp.pi * (g_coeffs[0] + (2 / 3) * g_coeffs[1]))
    rng = np.random.default_rng(31)
    worst_term = 0.0
    worst_lc = 0.0
    for _ in range(25):
        state = _make_state(rng, n_times=5)
        lc = ringed_lightcurve(state)
        p_para, o_para, i_para = _per_timestep_geometry(state)
        st = dict(state)
        setup = _lightcurve_setup(st, False)
        k = 2  # mid-transit timestep
        planet_para = {kk: v[k] for kk, v in p_para.items()}
        planet_two = {kk: v[k] for kk, v in setup["two"].items()}
        outer_para = {kk: v[k] for kk, v in o_para.items()}
        inner_para = {kk: v[k] for kk, v in i_para.items()}

        sh_p = _poly(planet_para)
        sh_o = _poly(outer_para)
        sh_i = _poly(inner_para)

        # pair and triple terms for the outer edge
        ps_xs, ps_ys = _single_intersection_points(**planet_two)
        rs = parametric_conic_intersections(**outer_para, **STAR_TWO)
        pr = parametric_conic_intersections(**outer_para, **planet_two)
        got_pair = float(ring_star_term(outer_para, *rs, g_coeffs) * norm * jnp.pi)
        got_triple = float(
            triple_term(planet_para, outer_para, ps_xs, ps_ys, *rs, *pr, g_coeffs)
            * norm
            * jnp.pi
        )
        worst_term = max(
            worst_term,
            abs(got_pair - sh_o.intersection(_STAR_POLY).area),
            abs(got_triple - sh_p.intersection(sh_o).intersection(_STAR_POLY).area),
        )

        # the fully-assembled blocked fraction
        blocked = sh_p.union(sh_o.difference(sh_i)).intersection(_STAR_POLY).area
        worst_lc = max(worst_lc, abs(float(1 - lc[k]) - blocked / np.pi))

    assert worst_term < AREA_TOL, f"worst term |area err| = {worst_term:.2e}"
    assert worst_lc < AREA_TOL, f"worst lightcurve |blocked err| = {worst_lc:.2e}"
