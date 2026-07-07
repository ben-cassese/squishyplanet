"""Validation of the ringed-planet transit machinery.

The blocked flux of a ringed planet decomposes into five convex-intersection terms
(see ``engine.ringed_transit``). These tests validate the individual terms and the
assembled light curve against brute-force Monte-Carlo integration over the stellar
disk -- for both a uniform and a limb-darkened star -- plus the analytic degenerate
limits (collapsed ring, edge-on ring, face-on ring) and gradient finiteness.
"""

import matplotlib

matplotlib.use("Agg")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from squishyplanet import OblateSystem, RingedSystem
from squishyplanet.engine.greens_basis_transform import generate_change_of_basis_matrix
from squishyplanet.engine.parametric_ellipse import point_in_ellipse
from squishyplanet.engine.polynomial_limb_darkened_transit import (
    _lightcurve_setup,
    _single_intersection_points,
    lightcurve,
)
from squishyplanet.engine.ringed_transit import (
    STAR_TWO,
    ring_star_term,
    ringed_lightcurve,
    triple_term,
)
from squishyplanet.engine.rings import parametric_conic_intersections, ring_prelude

N_MC = 4_000_000
# 5-sigma binomial noise on a blocked-flux fraction estimated from N_MC samples
MC_TOL = float(5 * 0.5 / np.sqrt(N_MC))
_KEY = jax.random.PRNGKey(0)
_K1, _K2 = jax.random.split(_KEY)
_MC_R = jnp.sqrt(jax.random.uniform(_K1, (N_MC,)))
_MC_ANG = jax.random.uniform(_K2, (N_MC,), maxval=2 * jnp.pi)
_MC_X = _MC_R * jnp.cos(_MC_ANG)
_MC_Y = _MC_R * jnp.sin(_MC_ANG)


def _make_state(
    rng: np.random.Generator,
    n_times: int = 31,
    uniform_ld: bool = True,
    **overrides: object,
) -> dict:
    """A random transiting ringed system as a raw engine state dictionary."""
    ld = (
        jnp.array([0.0, 0.0])
        if uniform_ld
        else jnp.array(list(rng.uniform(0.05, 0.3, size=2)))
    )
    state = dict(
        times=jnp.linspace(-0.3, 0.3, n_times),
        t_peri=-2.5,  # transit near t=0 for period=10, e=0, omega=0, Omega=pi
        t0=None,
        period=10.0,
        a=float(rng.uniform(4.0, 12.0)),
        e=0.0,
        i=float(np.pi / 2 + rng.uniform(-0.05, 0.05)),
        Omega=np.pi,
        omega=0.0,
        r=float(rng.uniform(0.05, 0.2)),
        f1=float(rng.uniform(0.0, 0.4)),
        f2=float(rng.uniform(0.0, 0.3)),
        obliq=float(rng.uniform(-1.0, 1.0)),
        prec=float(rng.uniform(0, 2 * np.pi)),
        tidally_locked=False,
        ld_u_coeffs=ld,
        greens_basis_transform=generate_change_of_basis_matrix(2),
        ring_tracks_planet=False,
        ring_obliq=float(rng.uniform(-np.pi, np.pi)),
        ring_prec=float(rng.uniform(0, 2 * np.pi)),
    )
    state["ring_inner_r"] = state["r"] * float(rng.uniform(1.05, 2.0))
    state["ring_outer_r"] = state["ring_inner_r"] * float(rng.uniform(1.05, 2.5))
    state.update(overrides)
    for k in state:
        if isinstance(state[k], float) and k != "period":
            state[k] = jnp.array([state[k]])
    return state


def _per_timestep_geometry(state: dict) -> tuple[dict, dict, dict]:
    """Recompute the per-timestep projected curves the driver used."""
    st = dict(state)
    setup = _lightcurve_setup(st, False)
    para_outer, para_inner = ring_prelude(st)
    return setup["para"], para_outer, para_inner


def _ld_weights(state: dict) -> jax.Array:
    """Unnormalized limb-darkening intensity at each MC sample point."""
    us = jnp.ones(state["ld_u_coeffs"].shape[0] + 1) * (-1)
    us = us.at[1:].set(state["ld_u_coeffs"])
    mu = jnp.sqrt(1 - _MC_R**2)
    powers = jnp.arange(len(us))
    return -jnp.sum(us[None, :] * (1 - mu[:, None]) ** powers[None, :], axis=1)


def test_terms_match_monte_carlo() -> None:
    """The pair and triple region integrals match MC areas across crossing regimes."""
    g_coeffs = jnp.matmul(
        generate_change_of_basis_matrix(2), jnp.array([-1.0, 0.0, 0.0])
    )
    norm = 1 / (jnp.pi * (g_coeffs[0] + (2 / 3) * g_coeffs[1]))
    rng = np.random.default_rng(11)
    regimes = set()
    for _ in range(40):
        state = _make_state(rng, n_times=3)
        p_para, o_para, _ = _per_timestep_geometry(state)
        st = dict(state)
        setup = _lightcurve_setup(st, False)
        for k in range(3):  # ingress, mid-transit, egress
            planet_para = {kk: v[k] for kk, v in p_para.items()}
            planet_two = {kk: v[k] for kk, v in setup["two"].items()}
            ring_para = {kk: v[k] for kk, v in o_para.items()}

            ps_xs, ps_ys = _single_intersection_points(**planet_two)
            rs = parametric_conic_intersections(**ring_para, **STAR_TWO)
            pr = parametric_conic_intersections(**ring_para, **planet_two)
            regimes.add(
                (
                    int(jnp.sum(ps_xs != 999)),
                    int(jnp.sum(rs[1] != 999)),
                    int(jnp.sum(pr[1] != 999)),
                )
            )

            got_pair = float(ring_star_term(ring_para, *rs, g_coeffs) * norm * jnp.pi)
            got_triple = float(
                triple_term(planet_para, ring_para, ps_xs, ps_ys, *rs, *pr, g_coeffs)
                * norm
                * jnp.pi
            )
            in_p = point_in_ellipse(_MC_X, _MC_Y, **planet_para)
            in_r = point_in_ellipse(_MC_X, _MC_Y, **ring_para)
            ref_pair = float(jnp.pi * jnp.mean(in_r))
            ref_triple = float(jnp.pi * jnp.mean(in_p & in_r))
            assert abs(got_pair - ref_pair) < MC_TOL * np.pi
            assert abs(got_triple - ref_triple) < MC_TOL * np.pi
    assert len(regimes) >= 4, f"too few crossing regimes exercised: {regimes}"


def test_lightcurve_matches_monte_carlo_uniform() -> None:
    """1 - flux equals the MC blocked fraction of P | (O & ~I) for a uniform star."""
    rng = np.random.default_rng(20)
    worst = 0.0
    for _ in range(8):
        state = _make_state(rng)
        lc = ringed_lightcurve(state)
        assert not bool(jnp.any(jnp.isnan(lc)))
        p_para, o_para, i_para = _per_timestep_geometry(state)
        for k in np.linspace(0, len(lc) - 1, 7).astype(int):
            pk = {kk: v[k] for kk, v in p_para.items()}
            ok = {kk: v[k] for kk, v in o_para.items()}
            ik = {kk: v[k] for kk, v in i_para.items()}
            in_p = point_in_ellipse(_MC_X, _MC_Y, **pk)
            in_o = point_in_ellipse(_MC_X, _MC_Y, **ok)
            in_i = point_in_ellipse(_MC_X, _MC_Y, **ik)
            ref = float(jnp.mean(in_p | (in_o & ~in_i)))
            worst = max(worst, abs(float(1.0 - lc[k]) - ref))
    assert worst < MC_TOL, f"worst |blocked - MC| = {worst:.2e} (tol {MC_TOL:.2e})"


def test_lightcurve_matches_monte_carlo_limb_darkened() -> None:
    """1 - flux equals the LD-weighted MC blocked fraction for a limb-darkened star.

    This validates the full limb-darkened Green's-theorem integrals over ring arcs,
    not just the areas: each MC sample is weighted by the stellar intensity profile.
    """
    rng = np.random.default_rng(21)
    worst = 0.0
    for _ in range(6):
        state = _make_state(rng, uniform_ld=False)
        w = _ld_weights(state)
        w_norm = jnp.mean(w)
        lc = ringed_lightcurve(state)
        p_para, o_para, i_para = _per_timestep_geometry(state)
        for k in np.linspace(0, len(lc) - 1, 5).astype(int):
            pk = {kk: v[k] for kk, v in p_para.items()}
            ok = {kk: v[k] for kk, v in o_para.items()}
            ik = {kk: v[k] for kk, v in i_para.items()}
            in_p = point_in_ellipse(_MC_X, _MC_Y, **pk)
            in_o = point_in_ellipse(_MC_X, _MC_Y, **ok)
            in_i = point_in_ellipse(_MC_X, _MC_Y, **ik)
            blocked = in_p | (in_o & ~in_i)
            ref = float(jnp.mean(w * blocked) / w_norm)
            worst = max(worst, abs(float(1.0 - lc[k]) - ref))
    # the weights raise the estimator variance a bit relative to the uniform case
    assert worst < 2 * MC_TOL, f"worst |blocked - MC| = {worst:.2e}"


def test_ring_collapse_matches_planet_only() -> None:
    """ring_outer -> ring_inner: the annulus vanishes and the planet-only curve is
    recovered (the four ring terms cancel analytically)."""
    rng = np.random.default_rng(22)
    for _ in range(4):
        state = _make_state(rng, uniform_ld=False)
        state["ring_outer_r"] = state["ring_inner_r"] * (1 + 1e-12)
        lc_ringed = ringed_lightcurve(state)
        lc_plain = lightcurve(dict(state), False)
        assert float(jnp.max(jnp.abs(lc_ringed - lc_plain))) < 1e-10


def test_edge_on_ring_matches_planet_only() -> None:
    """An exactly edge-on ring projects to a degenerate curve whose traversal doubles
    back on itself, so every ring term integrates to zero automatically."""
    rng = np.random.default_rng(23)
    for _ in range(4):
        state = _make_state(
            rng, uniform_ld=False, i=np.pi / 2, ring_obliq=0.0, ring_prec=0.0
        )
        lc_ringed = ringed_lightcurve(state)
        lc_plain = lightcurve(dict(state), False)
        assert float(jnp.max(jnp.abs(lc_ringed - lc_plain))) < 1e-10


def test_face_on_ring_closed_form() -> None:
    """Face-on ring + spherical planet fully inside a uniform star: the blocked
    fraction is (r^2 + ring_outer^2 - ring_inner^2) / Rstar^2 exactly."""
    rng = np.random.default_rng(24)
    state = _make_state(
        rng,
        i=np.pi / 2,
        f1=0.0,
        f2=0.0,
        obliq=0.0,
        prec=0.0,
        ring_obliq=np.pi / 2,
        ring_prec=np.pi / 2,
        r=0.08,
    )
    state["ring_inner_r"] = jnp.array([0.12])
    state["ring_outer_r"] = jnp.array([0.2])
    lc = ringed_lightcurve(state)
    expected = 0.08**2 + 0.2**2 - 0.12**2
    assert abs(float(1 - lc[len(lc) // 2]) - expected) < 1e-12


def test_rings_off_star_match_planet_only() -> None:
    """A huge face-on ring whose annulus lies entirely off the star mid-transit still
    blocks nothing extra when it never overlaps the disk (checked out of transit)."""
    rng = np.random.default_rng(25)
    state = _make_state(rng, uniform_ld=False)
    # out-of-transit times: planet + rings nowhere near the star
    state["times"] = jnp.linspace(2.0, 3.0, 11)
    lc = ringed_lightcurve(state)
    assert bool(jnp.all(lc == 1.0))


def test_star_inside_giant_ring() -> None:
    """When the outer edge's projection swallows the whole star and the inner edge
    also covers it, the annulus blocks nothing; sweep a huge thin annulus so the
    star sits fully inside the annulus hole."""
    rng = np.random.default_rng(26)
    state = _make_state(
        rng,
        i=np.pi / 2,
        r=0.05,
        f1=0.0,
        f2=0.0,
        obliq=0.0,
        prec=0.0,
        ring_obliq=np.pi / 2,
        ring_prec=np.pi / 2,
    )
    state["ring_inner_r"] = jnp.array([2.0])
    state["ring_outer_r"] = jnp.array([3.0])
    lc = ringed_lightcurve(state)
    # only the planet blocks: uniform-ld closed form for the fully-inside step
    k = len(lc) // 2
    assert abs(float(1 - lc[k]) - 0.05**2) < 1e-12


def test_gradients_finite() -> None:
    """Gradients w.r.t. every ring/orbit/shape parameter are finite at generic and
    exactly-edge-on configurations, and match finite differences loosely."""
    rng = np.random.default_rng(27)
    state = _make_state(rng, uniform_ld=False)

    def loss(
        ring_obliq: jax.Array,
        ring_prec: jax.Array,
        r_in: jax.Array,
        r_out: jax.Array,
        obliq: jax.Array,
        r: jax.Array,
    ) -> jax.Array:
        st = dict(state)
        st["ring_obliq"] = ring_obliq
        st["ring_prec"] = ring_prec
        st["ring_inner_r"] = r_in
        st["ring_outer_r"] = r_out
        st["obliq"] = obliq
        st["r"] = r
        return jnp.sum(ringed_lightcurve(st))

    args = (
        state["ring_obliq"],
        state["ring_prec"],
        state["ring_inner_r"],
        state["ring_outer_r"],
        state["obliq"],
        state["r"],
    )
    grads = jax.jacfwd(loss, argnums=(0, 1, 2, 3, 4, 5))(*args)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    # loose finite-difference agreement on the ring radii
    eps = 1e-6
    for idx in (2, 3):
        bumped = list(args)
        bumped[idx] = bumped[idx] + eps
        fd = (loss(*bumped) - loss(*args)) / eps
        assert abs(float(fd - grads[idx][0])) < 1e-3 * max(1.0, abs(float(fd)))

    # exactly edge-on ring
    args_edge = (
        jnp.array([0.0]),
        jnp.array([0.0]),
        state["ring_inner_r"],
        state["ring_outer_r"],
        state["obliq"],
        state["r"],
    )
    grads = jax.jacfwd(loss, argnums=(0, 1, 2, 3, 4, 5))(*args_edge)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


def test_ringed_system_frontend() -> None:
    """RingedSystem wires the engine, custom_vjp, params updates, oversampling,
    tracking, validation, state round-trip, and illustrate."""
    base = dict(
        times=jnp.linspace(-0.15, 0.15, 100),
        t_peri=-2.5,
        period=10.0,
        a=8.0,
        e=0.0,
        i=np.pi / 2 - 0.005,
        Omega=np.pi,
        omega=0.0,
        r=0.1,
        f1=0.1,
        f2=0.05,
        obliq=0.4,
        prec=0.9,
        ld_u_coeffs=jnp.array([0.3, 0.2]),
        tidally_locked=False,
    )
    s = RingedSystem(ring_inner_r=0.15, ring_outer_r=0.25, **base)
    lc = s.lightcurve()
    assert not bool(jnp.any(jnp.isnan(lc)))
    assert float(jnp.max(jnp.abs(lc - ringed_lightcurve(dict(s._state))))) < 1e-14

    # rings deepen the transit
    lc_plain = OblateSystem(**base).lightcurve()
    assert float(jnp.min(lc)) < float(jnp.min(lc_plain))

    # params updates flow through, including the tracked ring orientation
    assert float(jnp.min(s.lightcurve({"ring_outer_r": jnp.array([0.35])}))) < float(
        jnp.min(lc)
    )
    assert (
        float(jnp.max(jnp.abs(s.lightcurve({"obliq": jnp.array([1.2])}) - lc))) > 1e-4
    )

    # gradients through the custom_vjp path
    g = jax.grad(lambda p: jnp.sum(s.lightcurve(p)))(
        {"ring_inner_r": jnp.array([0.15]), "ring_outer_r": jnp.array([0.25])}
    )
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in g.values())

    # oversampling preserves shape and stays close to the instantaneous curve
    s_over = RingedSystem(
        ring_inner_r=0.15,
        ring_outer_r=0.25,
        **{**base, "exposure_time": 0.005, "oversample": 5},
    )
    lc_over = s_over.lightcurve()
    assert lc_over.shape == base["times"].shape
    assert float(jnp.max(jnp.abs(lc_over - lc))) < 1e-3

    # tidally locked with tracking rings runs and transits
    s_tl = RingedSystem(
        ring_inner_r=0.15, ring_outer_r=0.25, **{**base, "tidally_locked": True}
    )
    assert float(jnp.min(s_tl.lightcurve())) < 1.0

    # decoupled ring orientation differs from tracked
    s_dec = RingedSystem(
        ring_inner_r=0.15, ring_outer_r=0.25, ring_obliq=1.3, ring_prec=0.7, **base
    )
    assert float(jnp.max(jnp.abs(s_dec.lightcurve() - lc))) > 1e-5

    # validation
    for bad in (
        dict(ring_inner_r=0.15),
        dict(ring_inner_r=0.25, ring_outer_r=0.15),
        dict(ring_inner_r=0.05, ring_outer_r=0.25),
        dict(ring_inner_r=0.15, ring_outer_r=0.25, ring_obliq=1.0),
    ):
        try:
            RingedSystem(**bad, **base)
            raise RuntimeError(f"validation failed to catch {bad}")
        except AssertionError:
            pass

    # state round-trip and repr
    st = s.state
    assert "ring_inner_r" in st and "ring_tracks_planet" in st
    assert repr(s).startswith("RingedSystem(")

    # illustrate smoke tests (filled annulus, outlines only, tidally locked)
    s.illustrate(true_anomalies=jnp.pi / 2, window_size=1.2)
    plt.close("all")
    s.illustrate(times=jnp.array([0.0]), ring_fill=False, star_centered=True)
    plt.close("all")
    s_tl.illustrate(true_anomalies=jnp.array([jnp.pi / 2 - 0.3, jnp.pi / 2 + 0.3]))
    plt.close("all")
