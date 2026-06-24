"""Stress tests for numerically delicate transit geometries.

The transit engine in
``squishyplanet.engine.polynomial_limb_darkened_transit.lightcurve`` switches between
several branches as the planet moves across the star (fully-out, straddling-the-limb,
fully-in), solves a quartic for the limb crossings, and integrates the Green's-theorem
solution vectors with two different quadrature rules. Each of those seams, plus a handful
of "special" input angles and degenerate shapes, is a place where a bug could surface as
either a non-finite flux (NaN/Inf) or a spurious discontinuity (a "spike") in an
otherwise smooth light curve.

These tests do *not* compare against an external ground truth (that is the job of
``test_against_jaxoplanet_transit`` and ``test_fast_transit_components``). Instead they
assert three model-internal invariants that any correct transit model must satisfy, over
a wide grid of pathological configurations:

1. **Finiteness / bounds.** Every flux value is finite and lies in ``[0, 1]`` (the
   blocked flux is non-negative and never exceeds the unobscured stellar flux). Note
   the upper bound ``flux <= 1`` only holds for a *physically valid* limb-darkening
   profile -- one with non-negative intensity ``I(mu) = 1 - sum_k u_k (1 - mu)^k >= 0``
   everywhere. If the profile dips negative near the limb (``sum_k u_k > 1``), a planet
   crossing that annulus correctly "blocks" negative intensity and the flux rises above
   1; that is expected behavior, not a bug, and is pinned separately in
   ``test_negative_limb_darkening_overshoots``. The sweeps/fuzz below therefore restrict
   themselves to valid coefficients (``u_k >= 0`` and ``sum_k u_k <= 1``, which is
   sufficient for ``I >= 0``).

2. **Continuity (no spikes).** A transit light curve is continuous in time -- it has
   slope kinks at the contact points, but never jumps in value. We detect a value
   discontinuity with a grid-refinement argument: the largest step between adjacent
   samples, ``max|diff(flux)||``, scales like ``(max slope) * dt`` for a continuous
   curve, so refining the time grid by 4x must shrink it by roughly 4x. A jump
   discontinuity of size ``J`` leaves ``max|diff|`` pinned near ``J`` regardless of
   ``dt``, so it fails to shrink. We flag any config whose max step does not fall by at
   least ~40% under 4x refinement. This is deliberately a *gross*-spike detector: the
   sub-ppm scheme-handoff continuity is pinned separately and far more tightly by
   ``test_fast_transit_components.test_scheme_handoff_is_continuous``.

3. **Finite gradients.** ``sqrt`` and ``arctan2`` at the grazing tangent have infinite
   derivatives, so a value can be finite while its gradient is NaN. We check that
   ``jacfwd`` of the light curve w.r.t. each orbital element stays finite for the
   spike-prone geometries.

All tests drive the public ``lightcurve`` entry point through ``OblateSystem`` states.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from squishyplanet import OblateSystem
from squishyplanet.engine.kepler import skypos
from squishyplanet.engine.polynomial_limb_darkened_transit import lightcurve

# ---------------------------------------------------------------------------
# fixed geometry so the transit window (and hence grid spacing) is predictable
# ---------------------------------------------------------------------------
SMA = 12.0  # semimajor axis [Rstar]
PERIOD = 8.0  # [days]
R_DEF = 0.1  # default planet radius [Rstar]

# grid sizes. FINE has exactly 4x the spacing of COARSE over the same window
# ((COARSE_N - 1) * 4 + 1), which is what the refinement-continuity test relies on.
COARSE_N = 2001
FINE_N = (COARSE_N - 1) * 4 + 1  # 8001
GRAD_N = 401  # smaller grid for the (more expensive) gradient checks
FUZZ_N = 501  # grid for the randomized finiteness sweep


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _incl_for_b(b, a=SMA):
    """Inclination [rad] giving (circular) impact parameter ``b = a cos(i)``."""
    return float(np.arccos(np.clip(b / a, -1.0, 1.0)))


def _ld_min_intensity(u):
    """Minimum of the polynomial LD profile ``I(mu) = 1 - sum_k u_k (1 - mu)^k``."""
    u = np.asarray(u)
    mu = np.linspace(0.0, 1.0, 1001)
    intensity = 1.0 - np.sum(
        [u[k] * (1.0 - mu) ** (k + 1) for k in range(u.shape[0])], axis=0
    )
    return float(intensity.min())


def _valid_ld(order, rng, total=0.8):
    """Random physically-valid LD coefficients: ``u_k >= 0`` and ``sum_k u_k = total``.

    ``u_k >= 0`` with ``sum_k u_k <= 1`` guarantees ``I(mu) >= 0`` over the whole disk
    (each ``(1 - mu)^k <= 1``), so the ``flux <= 1`` invariant holds.
    """
    raw = rng.uniform(0.0, 1.0, size=order)
    u = raw / raw.sum() * total
    assert _ld_min_intensity(u) >= -1e-12  # guard the helper itself
    return jnp.array(u)


def _transit_half_window(a, period, r, e):
    """Generous half-window [days] guaranteed to bracket ingress and egress.

    ~3x the small-angle transit half-duration, inflated by the eccentric
    speed-up/slow-down factor ``sqrt((1+e)/(1-e))`` so even high-e transits sit
    comfortably inside the window with flat baseline on either end.
    """
    half = (period / (2.0 * np.pi)) * (1.0 + r) / a * np.sqrt((1.0 + e) / (1.0 - e))
    return 3.0 * half


def _times(n, a=SMA, period=PERIOD, r=R_DEF, e=0.0):
    w = _transit_half_window(a, period, r, e)
    return jnp.linspace(-w, w, n)


def _build_state(times, **over):
    """Build an ``OblateSystem`` ``_state`` for the 3D-parameterization branch.

    Defaults give an edge-on, central, circular, quadratic-LD transit; ``over``
    replaces any field. ``omega`` is forced to 0 for circular orbits (the validator
    requires it).
    """
    base = dict(
        t0=0.0,
        period=PERIOD,
        a=SMA,
        e=0.0,
        i=jnp.pi / 2,
        Omega=jnp.pi,
        omega=0.0,
        obliq=0.0,
        prec=0.0,
        r=R_DEF,
        f1=0.0,
        f2=0.0,
        ld_u_coeffs=jnp.array([0.4, 0.3]),
        tidally_locked=False,
    )
    base.update(over)
    if float(base["e"]) == 0.0:
        base["omega"] = 0.0
    return OblateSystem(times=times, **base)._state


def _dense_lc(n, **over):
    """Light curve (numpy) over a window auto-sized to ``over``'s a/period/r/e."""
    a = float(over.get("a", SMA))
    period = float(over.get("period", PERIOD))
    r = float(over.get("r", R_DEF))
    e = float(over.get("e", 0.0))
    t = _times(n, a=a, period=period, r=r, e=e)
    state = _build_state(t, **over)
    return np.asarray(lightcurve(state, False))


# ---------------------------------------------------------------------------
# assertion helpers
# ---------------------------------------------------------------------------
def _assert_finite_bounded(flux, name):
    assert np.all(np.isfinite(flux)), f"{name}: non-finite flux value(s)"
    assert np.all(flux <= 1.0 + 1e-9), f"{name}: flux exceeds 1 (negative blocked flux)"
    assert np.all(flux >= -1e-9), f"{name}: flux below 0 (over-blocked)"


def _max_step(flux):
    return float(np.max(np.abs(np.diff(flux))))


def _assert_continuous(name, floor=1e-7, ratio=0.6, **over):
    """No value-discontinuity: the max adjacent step must shrink under 4x refinement.

    Skips the refinement check when the coarse max step is below ``floor`` (the curve is
    essentially flat -- a near/total miss -- so there is no feature to resolve and the
    step is numerical noise); finiteness/bounds are still asserted on both grids.
    """
    coarse = _dense_lc(COARSE_N, **over)
    fine = _dense_lc(FINE_N, **over)
    _assert_finite_bounded(coarse, f"{name} (coarse)")
    _assert_finite_bounded(fine, f"{name} (fine)")

    mc = _max_step(coarse)
    mf = _max_step(fine)
    if mc > floor:
        assert mf < ratio * mc, (
            f"{name}: suspected discontinuity -- max adjacent step failed to shrink "
            f"under 4x grid refinement (coarse={mc:.3e}, fine={mf:.3e}, "
            f"ratio={mf / mc:.2f} >= {ratio})"
        )


# ===========================================================================
# 1. Continuity (refinement) on curated spike-prone geometries
# ===========================================================================
# Each entry: (id, overrides). These are the configs where a branch handoff or quartic
# degeneracy is most likely to leak a discontinuity.
_CONTINUITY_CASES = [
    ("central_edge_on", dict(i=jnp.pi / 2)),
    ("mid_impact", dict(i=_incl_for_b(0.5))),
    ("grazing_b0.95", dict(i=_incl_for_b(0.95))),
    ("near_grazing_b0.99", dict(i=_incl_for_b(0.99))),
    ("deep_planet_r0.3", dict(r=0.3, i=_incl_for_b(0.5))),
    ("large_planet_r0.25_central", dict(r=0.25, i=jnp.pi / 2)),
    ("tiny_planet_r1e-3", dict(r=1e-3, i=jnp.pi / 2)),
    # oblate / triaxial -- exercises non-axis-aligned outlines and the sxx-syy guard
    ("oblate_axis_aligned", dict(f1=0.5, obliq=0.0, prec=0.0, i=_incl_for_b(0.5))),
    ("oblate_obliq45", dict(f1=0.5, obliq=jnp.pi / 4, i=_incl_for_b(0.5))),
    ("triaxial", dict(f1=0.4, f2=0.3, obliq=0.7, prec=1.1, i=_incl_for_b(0.6))),
    ("grazing_oblate", dict(f1=0.5, obliq=jnp.pi / 4, i=_incl_for_b(0.95))),
    ("pole_on_oblate", dict(f1=0.5, obliq=jnp.pi / 2, i=_incl_for_b(0.4))),
    # eccentric -- transit at periastron / off-quadrature arguments of periastron
    ("ecc_peri", dict(e=0.5, omega=jnp.pi / 2, i=_incl_for_b(0.3))),
    ("ecc_apo", dict(e=0.7, omega=3 * jnp.pi / 2, i=_incl_for_b(0.4))),
    ("ecc_oblique_omega", dict(e=0.6, omega=jnp.pi / 3, i=_incl_for_b(0.5))),
    # off-axis ascending node
    ("Omega_pi_over_3", dict(Omega=jnp.pi / 3, i=_incl_for_b(0.6))),
    ("Omega_zero", dict(Omega=0.0, i=_incl_for_b(0.6))),
    # higher-order limb darkening (stresses the (1 - x^2 - y^2)^(n/2) terms)
    (
        "ld_order_8",
        dict(
            ld_u_coeffs=jnp.array([0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]),
            i=_incl_for_b(0.4),
        ),
    ),
    ("uniform_disk", dict(ld_u_coeffs=jnp.array([0.0, 0.0]), i=_incl_for_b(0.5))),
]


@pytest.mark.parametrize(
    "name, over", _CONTINUITY_CASES, ids=[c[0] for c in _CONTINUITY_CASES]
)
def test_no_spikes_curated(name, over):
    """No NaNs, bounds violations, or value discontinuities in spike-prone geometries."""
    _assert_continuous(name, **over)


# ===========================================================================
# 2. Grazing / tangency sweeps (finite + bounded)
# ===========================================================================
def test_grazing_sweep_spherical():
    """Sweep the impact parameter straight through external tangency (b = 1 + r).

    This crosses every transit branch boundary: full miss -> straddling -> partial ->
    (for small enough b) fully-inside, including the exact-tangency points that are
    measure-zero in a real observation but reachable with crafted inputs.
    """
    for b in np.linspace(0.80, 1.20, 41):
        flux = _dense_lc(COARSE_N, i=_incl_for_b(b))
        _assert_finite_bounded(flux, f"grazing_sweep b={b:.3f}")


def test_grazing_sweep_oblate():
    """Same impact-parameter sweep for a flattened, tilted planet, where the grazing
    geometry depends on the projected ellipse orientation."""
    for b in np.linspace(0.80, 1.20, 41):
        flux = _dense_lc(
            COARSE_N, i=_incl_for_b(b), f1=0.5, f2=0.2, obliq=jnp.pi / 5, prec=0.8
        )
        _assert_finite_bounded(flux, f"grazing_sweep_oblate b={b:.3f}")


@pytest.mark.parametrize("b", [1.0 - R_DEF, 1.0, 1.0 + R_DEF])
def test_exact_tangency(b):
    """Exact internal tangency (b = 1 - r, the 2nd/3rd-contact scheme handoff), the
    center-on-limb case (b = 1), and exact external tangency (b = 1 + r)."""
    flux = _dense_lc(COARSE_N, i=_incl_for_b(b))
    _assert_finite_bounded(flux, f"exact_tangency b={b:.4f}")


def test_total_miss_stays_flat():
    """A planet that never reaches the limb (b well above 1 + r) returns flat unit flux."""
    flux = _dense_lc(COARSE_N, i=_incl_for_b(1.5))
    _assert_finite_bounded(flux, "total_miss")
    assert np.allclose(flux, 1.0, atol=1e-12), "non-transiting case is not flat at 1.0"


# ===========================================================================
# 3. Special-angle sweeps (finite + bounded)
# ===========================================================================
_SPECIAL_ANGLES = [
    0.0,
    np.pi / 6,
    np.pi / 4,
    np.pi / 3,
    np.pi / 2,
    2 * np.pi / 3,
    3 * np.pi / 4,
    np.pi,
    3 * np.pi / 2,
    2 * np.pi,
]


@pytest.mark.parametrize("ang", _SPECIAL_ANGLES)
def test_special_Omega(ang):
    """Ascending node at the 'nice' angles that can axis-align the projected ellipse."""
    flux = _dense_lc(COARSE_N, Omega=ang, i=_incl_for_b(0.5), f1=0.4, obliq=jnp.pi / 6)
    _assert_finite_bounded(flux, f"Omega={ang:.4f}")


@pytest.mark.parametrize("ang", _SPECIAL_ANGLES)
def test_special_omega(ang):
    """Argument of periastron at special angles (e > 0 so it is meaningful)."""
    flux = _dense_lc(COARSE_N, e=0.3, omega=ang, i=_incl_for_b(0.5))
    _assert_finite_bounded(flux, f"omega={ang:.4f}")


@pytest.mark.parametrize("ang", [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
def test_special_obliq(ang):
    """Planet obliquity at special angles for a flattened planet (axis-aligned ->
    tilted -> pole-on outline)."""
    flux = _dense_lc(COARSE_N, obliq=ang, f1=0.5, i=_incl_for_b(0.5))
    _assert_finite_bounded(flux, f"obliq={ang:.4f}")


@pytest.mark.parametrize("ang", _SPECIAL_ANGLES)
def test_special_prec(ang):
    """Planet precession angle at special angles for a triaxial planet."""
    flux = _dense_lc(COARSE_N, prec=ang, f1=0.4, f2=0.25, obliq=0.6, i=_incl_for_b(0.5))
    _assert_finite_bounded(flux, f"prec={ang:.4f}")


# ===========================================================================
# 4. Eccentricity / radius / flattening / limb-darkening sweeps (finite + bounded)
# ===========================================================================
@pytest.mark.parametrize("e", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
@pytest.mark.parametrize("omega", [np.pi / 2, np.pi, 3 * np.pi / 2])
def test_eccentricity_sweep(e, omega):
    flux = _dense_lc(
        COARSE_N, e=e, omega=(0.0 if e == 0.0 else omega), i=_incl_for_b(0.4)
    )
    _assert_finite_bounded(flux, f"e={e:.2f}, omega={omega:.3f}")


@pytest.mark.parametrize("r", [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.5])
def test_radius_sweep(r):
    flux = _dense_lc(COARSE_N, r=r, i=_incl_for_b(0.4))
    _assert_finite_bounded(flux, f"r={r:.4g}")


@pytest.mark.parametrize("f1", [0.0, 0.1, 0.3, 0.5, 0.8, 0.95])
@pytest.mark.parametrize("f2", [0.0, 0.3, 0.6])
def test_flattening_sweep(f1, f2):
    """Oblateness/triaxiality up to extreme values, at a tilted orientation."""
    flux = _dense_lc(
        COARSE_N, f1=f1, f2=f2, obliq=jnp.pi / 5, prec=0.7, i=_incl_for_b(0.4)
    )
    _assert_finite_bounded(flux, f"f1={f1}, f2={f2}")


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 8])
def test_ld_order_sweep(order):
    """Polynomial limb darkening of increasing order (physically valid profiles)."""
    rng = np.random.default_rng(100 + order)
    u = _valid_ld(order, rng)
    flux = _dense_lc(COARSE_N, ld_u_coeffs=u, i=_incl_for_b(0.4))
    _assert_finite_bounded(flux, f"ld_order={order}")


@pytest.mark.parametrize(
    "u",
    [
        jnp.array([0.0, 0.0]),  # uniform disk
        jnp.array([1.0, 0.0]),  # fully linear, edge -> 0 brightness
        jnp.array([0.0, 1.0]),  # pure quadratic
        jnp.array([-0.2, 0.4]),  # negative leading coefficient
        jnp.array([0.9, 0.05]),
    ],
)
def test_ld_special_coeffs(u):
    assert _ld_min_intensity(u) >= -1e-12, "test case is meant to be a valid LD profile"
    flux = _dense_lc(COARSE_N, ld_u_coeffs=u, i=_incl_for_b(0.5))
    _assert_finite_bounded(flux, f"ld_coeffs={np.asarray(u)}")


def test_negative_limb_darkening_overshoots():
    """Unphysical LD (negative intensity near the limb) makes flux > 1 -- by design.

    This is NOT a transit-model bug: if the limb-darkening polynomial dips below zero
    near the stellar limb (``sum_k u_k > 1`` so ``I(mu=0) < 0``), then a planet crossing
    that outermost annulus "blocks" negative intensity, which *adds* flux. The overshoot
    is therefore the model faithfully integrating a negative-intensity region, and it is
    localized to the contact phases where the planet overlaps only the limb.

    We pin three things so this stays understood and any future change is caught:
      - the flux genuinely exceeds 1 (the overshoot is real, not noise),
      - it remains finite and continuous (no NaN, no discontinuous spike), and
      - the overshoot occurs only while the planet straddles the limb
        (``|d_center - 1| <~ r``), never deep inside the disk.
    """
    order = 8
    u = jnp.array([0.5 / (k + 1) for k in range(order)])  # sum > 1 -> I(limb) < 0
    assert _ld_min_intensity(u) < -1e-3, "this case must have a negative-intensity limb"

    r = R_DEF
    over = dict(r=r, i=_incl_for_b(0.4), ld_u_coeffs=u)
    coarse = _dense_lc(COARSE_N, **over)
    fine = _dense_lc(FINE_N, **over)

    # finite + continuous (the algorithm is well-behaved; only the bound is violated)
    assert np.all(np.isfinite(coarse)) and np.all(np.isfinite(fine))
    mc, mf = _max_step(coarse), _max_step(fine)
    assert (
        mf < 0.6 * mc
    ), f"overshoot is a smooth feature, not a spike (mc={mc:.2e}, mf={mf:.2e})"

    # the overshoot is real and modest (tens of ppm here), not a blow-up
    overshoot = coarse.max() - 1.0
    assert 1e-6 < overshoot < 1e-2, f"unexpected overshoot magnitude {overshoot:.2e}"

    # and it lives only at the limb-crossing phases, not deep in transit
    state = _build_state(_times(COARSE_N, r=r), **over)
    pos = np.asarray(skypos(**state))
    d_center = np.hypot(pos[0], pos[1])
    above = coarse > 1.0 + 1e-9
    assert np.all(
        np.abs(d_center[above] - 1.0) < r + 0.05
    ), "flux exceeds 1 only where the planet straddles the stellar limb"


# ===========================================================================
# 5. Gradient finiteness on spike-prone geometries
# ===========================================================================
_GRAD_CASES = [
    ("grazing", dict(i=_incl_for_b(0.99))),
    ("internal_tangency", dict(i=_incl_for_b(1.0 - R_DEF))),
    ("grazing_oblate", dict(f1=0.5, obliq=jnp.pi / 4, i=_incl_for_b(0.97))),
    ("ecc_peri", dict(e=0.6, omega=jnp.pi / 2, i=_incl_for_b(0.3))),
    ("deep_central", dict(r=0.3, i=jnp.pi / 2)),
]

_GRAD_PARAMS = ["i", "r", "e", "omega", "obliq", "f1", "a"]


@pytest.mark.parametrize("name, over", _GRAD_CASES, ids=[c[0] for c in _GRAD_CASES])
def test_gradients_finite(name, over):
    """jacfwd of the light curve w.r.t each orbital element stays finite.

    The grazing tangent is where sqrt/arctan2 have infinite derivatives, so this is the
    natural place for a finite-valued-but-NaN-gradient bug to hide.
    """
    a = float(over.get("a", SMA))
    period = float(over.get("period", PERIOD))
    r = float(over.get("r", R_DEF))
    e = float(over.get("e", 0.0))
    t = _times(GRAD_N, a=a, period=period, r=r, e=e)
    state = _build_state(t, **over)

    for key in _GRAD_PARAMS:
        # skip omega for circular orbits (it is degenerate / pinned to 0)
        if key == "omega" and float(state["e"][0]) == 0.0:
            continue

        def f(x, _key=key):
            s = dict(state)
            s[_key] = x
            return lightcurve(s, False)

        J = jax.jacfwd(f)(state[key])
        assert np.all(
            np.isfinite(np.asarray(J))
        ), f"{name}: non-finite d(flux)/d({key})"


# ===========================================================================
# 6. Randomized fuzz sweep (finite + bounded)
# ===========================================================================
def test_random_pathological_fuzz():
    """Many random configurations spanning the full pathological input space.

    Covers simultaneous combinations (eccentric + triaxial + tilted + high-order LD +
    grazing) that the structured sweeps above test only one axis at a time. Asserts only
    finiteness/bounds (cheap), so we can afford a large sample.
    """
    rng = np.random.default_rng(20240624)
    n_cfg = 200
    tp = 2 * np.pi
    for k in range(n_cfg):
        e = float(rng.uniform(0.0, 0.9))
        b = float(rng.uniform(0.0, 1.15))
        order = int(rng.integers(2, 7))
        over = dict(
            e=e,
            omega=(0.0 if e == 0.0 else float(rng.uniform(0, tp))),
            Omega=float(rng.uniform(0, tp)),
            i=_incl_for_b(b),
            obliq=float(rng.uniform(-np.pi / 2, np.pi / 2)),
            prec=float(rng.uniform(0, tp)),
            r=float(rng.uniform(0.01, 0.3)),
            f1=float(rng.uniform(0.0, 0.7)),
            f2=float(rng.uniform(0.0, 0.5)),
            ld_u_coeffs=_valid_ld(order, rng, total=float(rng.uniform(0.2, 1.0))),
        )
        flux = _dense_lc(FUZZ_N, **over)
        _assert_finite_bounded(flux, f"fuzz[{k}] b={b:.3f} e={e:.2f}")


# ===========================================================================
# 7. Projected-ellipse parameterization branch
# ===========================================================================
def _dense_lc_projected(n, projected_effective_r=0.1, **over):
    a = float(over.get("a", SMA))
    period = float(over.get("period", PERIOD))
    e = float(over.get("e", 0.0))
    t = _times(n, a=a, period=period, r=projected_effective_r, e=e)
    base = dict(
        t0=0.0,
        period=PERIOD,
        a=SMA,
        e=0.0,
        i=jnp.pi / 2,
        Omega=jnp.pi,
        omega=0.0,
        parameterize_with_projected_ellipse=True,
        projected_effective_r=projected_effective_r,
        projected_f=0.0,
        projected_theta=0.0,
        ld_u_coeffs=jnp.array([0.4, 0.3]),
        tidally_locked=False,
    )
    base.update(over)
    if float(base["e"]) == 0.0:
        base["omega"] = 0.0
    state = OblateSystem(times=t, **base)._state
    return np.asarray(lightcurve(state, True))


@pytest.mark.parametrize("pf", [0.0, 0.2, 0.5, 0.8, 0.95])
def test_projected_flattening_sweep(pf):
    """Projected-ellipse branch with flattening up to a near-degenerate sliver.

    As ``projected_f -> 1`` the (area-preserving) minor axis collapses toward zero,
    stressing the divisions in ``parameterize_2d_helper``.
    """
    flux = _dense_lc_projected(
        COARSE_N, projected_f=pf, projected_theta=jnp.pi / 3, i=_incl_for_b(0.4)
    )
    _assert_finite_bounded(flux, f"projected_f={pf}")


@pytest.mark.parametrize("ang", _SPECIAL_ANGLES)
def test_projected_theta_special(ang):
    """Projected-ellipse orientation at the 'nice' angles (axis-aligned outlines)."""
    flux = _dense_lc_projected(
        COARSE_N, projected_f=0.4, projected_theta=ang, i=_incl_for_b(0.4)
    )
    _assert_finite_bounded(flux, f"projected_theta={ang:.4f}")


def test_projected_branch_no_spikes():
    """Refinement-continuity check for the projected-ellipse branch (grazing, tilted)."""
    over = dict(projected_f=0.5, projected_theta=jnp.pi / 4, i=_incl_for_b(0.95))
    coarse = _dense_lc_projected(COARSE_N, **over)
    fine = _dense_lc_projected(FINE_N, **over)
    _assert_finite_bounded(coarse, "projected_no_spikes (coarse)")
    _assert_finite_bounded(fine, "projected_no_spikes (fine)")
    mc, mf = _max_step(coarse), _max_step(fine)
    if mc > 1e-7:
        assert mf < 0.6 * mc, (
            f"projected branch: suspected discontinuity (coarse={mc:.3e}, "
            f"fine={mf:.3e})"
        )
