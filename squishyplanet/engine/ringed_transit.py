"""Green's-theorem machinery for transits of ringed planets.

The blocked flux of a planet P with an opaque annular ring (inner/outer edge
ellipses I and O, with I inside O since the projected edges are concentric similar
ellipses) in front of the star S decomposes exactly, for any surface brightness, as

.. math::

    F(B) = F(P \\cap S) + F(O \\cap S) - F(I \\cap S)
           - F(P \\cap O \\cap S) + F(P \\cap I \\cap S)

Every term is an intersection of 2-3 convex regions, hence connected, and its
boundary is exactly the set of arcs of each bounding curve that lie inside all the
other regions. Each term is therefore computed with the same "exploratory" recipe as
:func:`polynomial_limb_darkened_transit.ellipse_star_term`: find all pairwise
crossings, split every curve into arcs at its crossings, keep the arcs whose
midpoints lie inside the other regions, and sum their Green's-theorem contributions
(all curves are traversed counterclockwise, so the summation order is irrelevant).

Ring edges can be arbitrarily close to degenerate (edge-on), where their implicit
conic coefficients diverge as the inverse squared semi-minor axis. All ring
intersections are therefore solved with the ring on the *parametric* side
(:func:`rings.parametric_conic_intersections` against the planet's or star's
well-conditioned implicit form), and all membership tests use the division-free
:func:`parametric_ellipse.point_in_ellipse`.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from squishyplanet.engine.parametric_ellipse import (
    cartesian_intersection_to_parametric_angle,
    point_in_ellipse,
)
from squishyplanet.engine.polynomial_limb_darkened_transit import (
    _arcs_from_angles,
    _lightcurve_setup,
    _s0_definite,
    _s1_definite,
    _single_intersection_points,
    ellipse_bound,
    ellipse_star_term,
    planet_solution_vec,
)
from squishyplanet.engine.rings import parametric_conic_intersections, ring_prelude

# the star as an implicit conic (x^2 + y^2 = 1), for parametric_conic_intersections
STAR_TWO = {
    "rho_xx": 1.0,
    "rho_xy": 0.0,
    "rho_x0": 0.0,
    "rho_yy": 1.0,
    "rho_y0": 0.0,
    "rho_00": 0.0,
}


def _normalize_angles(alphas: jax.Array, xs: jax.Array) -> jax.Array:
    """Map crossing angles into [0, 2 pi], sending sentinel slots to 2 pi.

    Args:
        alphas (Array): Raw parametric angles (any real value; 999 for sentinels).
        xs (Array): The matching x-coordinates, used only to spot 999 sentinels.

    Returns:
        Array: Angles in ``[0, 2 pi]`` ready for :func:`_arcs_from_angles`.

    """
    alphas = jnp.where(xs != 999, alphas, 2 * jnp.pi)
    alphas = jnp.where(alphas < 0, alphas + 2 * jnp.pi, alphas)
    alphas = jnp.where(alphas > 2 * jnp.pi, alphas - 2 * jnp.pi, alphas)
    return alphas


def _star_polar_angles(xs: jax.Array, ys: jax.Array) -> jax.Array:
    """Polar angles on the stellar limb of the given crossing points.

    Args:
        xs (Array): Crossing x-coordinates (999 for sentinels).
        ys (Array): Crossing y-coordinates (999 for sentinels).

    Returns:
        Array: Polar angles in ``[0, 2 pi]``, sentinels mapped to 2 pi.

    """
    thetas = jnp.where(xs != 999, jnp.arctan2(ys, xs), 2 * jnp.pi)
    return jnp.where(thetas < 0, thetas + 2 * jnp.pi, thetas)


def _gated_ellipse_arcs(
    para: dict, lo: jax.Array, hi: jax.Array, keep: jax.Array, g_coeffs: jax.Array
) -> jax.Array:
    """Sum the Green's-theorem contributions of the kept arcs of one ellipse.

    Each kept arc costs one fixed-grid quadrature; skipped arcs cost nothing thanks to
    the ``lax.cond`` inside the sequential scan.

    Args:
        para (dict): Parametric coefficients of the ellipse (single timestep).
        lo (Array): Arc lower bounds from :func:`_arcs_from_angles`.
        hi (Array): Arc upper bounds.
        keep (Array): Boolean, which arcs lie on the region boundary.
        g_coeffs (Array): Green's-basis limb darkening coefficients.

    Returns:
        Array: Scalar sum of ``planet_solution_vec(...) @ g_coeffs`` over kept arcs.

    """

    def body(carry: jax.Array, arc: tuple) -> tuple[jax.Array, None]:
        arc_lo, arc_hi, arc_keep = arc
        val = jax.lax.cond(
            arc_keep,
            lambda _: jnp.matmul(
                planet_solution_vec(arc_lo, arc_hi, g_coeffs, **para), g_coeffs
            ),
            lambda _: 0.0,
            None,
        )
        return carry + val, None

    return jax.lax.scan(body, 0.0, (lo, hi, keep))[0]


def _in_star(x: jax.Array, y: jax.Array) -> jax.Array:
    """Whether the point(s) are strictly inside the unit stellar disk."""
    return x**2 + y**2 < 1.0


def ring_star_term(
    ring_para: dict,
    rs_alphas: jax.Array,
    rs_xs: jax.Array,
    rs_ys: jax.Array,
    g_coeffs: jax.Array,
) -> jax.Array:
    """The blocked-flux boundary integral F(E intersect S) for one ring edge.

    The ring-edge analog of
    :func:`polynomial_limb_darkened_transit.ellipse_star_term`, except the
    ring-star crossings are precomputed by the caller (with
    :func:`rings.parametric_conic_intersections` against :data:`STAR_TWO`, keeping the
    possibly-degenerate ring on the parametric side) and shared with the triple term.

    Args:
        ring_para (dict): Parametric coefficients of the ring edge (single timestep,
            counterclockwise).
        rs_alphas (Array): Ring-side parametric angles of the ring-star crossings.
        rs_xs (Array): Crossing x-coordinates (999 sentinels).
        rs_ys (Array): Crossing y-coordinates.
        g_coeffs (Array): Green's-basis limb darkening coefficients.

    Returns:
        Array: Scalar blocked flux from this ring edge, before normalization.

    """
    # arcs of the ring edge, kept where inside the star
    lo, hi, mid = _arcs_from_angles(_normalize_angles(rs_alphas, rs_xs))
    mx = ring_para["c_x1"] * jnp.cos(mid) + ring_para["c_x2"] * jnp.sin(mid)
    mx = mx + ring_para["c_x3"]
    my = ring_para["c_y1"] * jnp.cos(mid) + ring_para["c_y2"] * jnp.sin(mid)
    my = my + ring_para["c_y3"]
    keep = _in_star(mx, my) & (hi > lo)
    ring_contribution = _gated_ellipse_arcs(ring_para, lo, hi, keep, g_coeffs)

    # arcs of the stellar limb, kept where inside the ring edge
    star_lo, star_hi, star_mid = _arcs_from_angles(_star_polar_angles(rs_xs, rs_ys))
    keep_star = point_in_ellipse(jnp.cos(star_mid), jnp.sin(star_mid), **ring_para) & (
        star_hi > star_lo
    )
    s0 = _s0_definite(star_lo, star_hi)
    s1 = _s1_definite(star_lo, star_hi)
    star_contribution = jnp.sum(
        jnp.where(keep_star, g_coeffs[0] * s0 + g_coeffs[1] * s1, 0.0)
    )

    return ring_contribution + star_contribution


def triple_term(
    planet_para: dict,
    ring_para: dict,
    ps_xs: jax.Array,
    ps_ys: jax.Array,
    rs_alphas: jax.Array,
    rs_xs: jax.Array,
    rs_ys: jax.Array,
    pr_alphas: jax.Array,
    pr_xs: jax.Array,
    pr_ys: jax.Array,
    g_coeffs: jax.Array,
) -> jax.Array:
    """The blocked-flux boundary integral F(P intersect E intersect S).

    The three-region term of the ringed-planet inclusion-exclusion decomposition, for
    the planet P, one ring edge E, and the star S. All three pairwise crossing sets
    are precomputed by the caller (each pairwise quartic is solved once per timestep
    and shared across the decomposition's terms). Each of the three curves is split
    into up to 9 arcs at its (up to 8) crossings with the other two curves, and an arc
    contributes exactly when its midpoint lies inside both other regions. Containment
    cases (e.g. P entirely inside E) fall out of the same code path via the bookended
    ``[0, 2 pi]`` arc, whose midpoint doubles as a whole-curve containment test.

    Args:
        planet_para (dict): Parametric coefficients of the planet outline
            (counterclockwise; single timestep).
        ring_para (dict): Parametric coefficients of the ring edge (counterclockwise).
        ps_xs (Array): Planet-star crossing x-coordinates (999 sentinels), from
            :func:`polynomial_limb_darkened_transit._single_intersection_points`.
        ps_ys (Array): Planet-star crossing y-coordinates.
        rs_alphas (Array): Ring-side angles of the ring-star crossings, from
            :func:`rings.parametric_conic_intersections` against :data:`STAR_TWO`.
        rs_xs (Array): Ring-star crossing x-coordinates.
        rs_ys (Array): Ring-star crossing y-coordinates.
        pr_alphas (Array): Ring-side angles of the planet-ring crossings, from
            :func:`rings.parametric_conic_intersections` against the planet's
            implicit conic.
        pr_xs (Array): Planet-ring crossing x-coordinates.
        pr_ys (Array): Planet-ring crossing y-coordinates.
        g_coeffs (Array): Green's-basis limb darkening coefficients.

    Returns:
        Array: Scalar blocked flux of the triple-intersection region, before
        normalization.

    """
    # --- arcs of the planet outline, kept where inside both the star and the ring
    ps_alphas_p = _normalize_angles(
        cartesian_intersection_to_parametric_angle(ps_xs, ps_ys, **planet_para), ps_xs
    )
    pr_alphas_p = _normalize_angles(
        cartesian_intersection_to_parametric_angle(pr_xs, pr_ys, **planet_para), pr_xs
    )
    lo, hi, mid = _arcs_from_angles(jnp.concatenate((ps_alphas_p, pr_alphas_p)))
    mx = planet_para["c_x1"] * jnp.cos(mid) + planet_para["c_x2"] * jnp.sin(mid)
    mx = mx + planet_para["c_x3"]
    my = planet_para["c_y1"] * jnp.cos(mid) + planet_para["c_y2"] * jnp.sin(mid)
    my = my + planet_para["c_y3"]
    keep = _in_star(mx, my) & point_in_ellipse(mx, my, **ring_para) & (hi > lo)
    planet_contribution = _gated_ellipse_arcs(planet_para, lo, hi, keep, g_coeffs)

    # --- arcs of the ring edge, kept where inside both the star and the planet
    rs_alphas_r = _normalize_angles(rs_alphas, rs_xs)
    pr_alphas_r = _normalize_angles(pr_alphas, pr_xs)
    lo, hi, mid = _arcs_from_angles(jnp.concatenate((rs_alphas_r, pr_alphas_r)))
    rx = ring_para["c_x1"] * jnp.cos(mid) + ring_para["c_x2"] * jnp.sin(mid)
    rx = rx + ring_para["c_x3"]
    ry = ring_para["c_y1"] * jnp.cos(mid) + ring_para["c_y2"] * jnp.sin(mid)
    ry = ry + ring_para["c_y3"]
    keep = _in_star(rx, ry) & point_in_ellipse(rx, ry, **planet_para) & (hi > lo)
    ring_contribution = _gated_ellipse_arcs(ring_para, lo, hi, keep, g_coeffs)

    # --- arcs of the stellar limb, kept where inside both the planet and the ring
    thetas = jnp.concatenate(
        (_star_polar_angles(ps_xs, ps_ys), _star_polar_angles(rs_xs, rs_ys))
    )
    star_lo, star_hi, star_mid = _arcs_from_angles(thetas)
    sx, sy = jnp.cos(star_mid), jnp.sin(star_mid)
    keep_star = (
        point_in_ellipse(sx, sy, **planet_para)
        & point_in_ellipse(sx, sy, **ring_para)
        & (star_hi > star_lo)
    )
    s0 = _s0_definite(star_lo, star_hi)
    s1 = _s1_definite(star_lo, star_hi)
    star_contribution = jnp.sum(
        jnp.where(keep_star, g_coeffs[0] * s0 + g_coeffs[1] * s1, 0.0)
    )

    return planet_contribution + ring_contribution + star_contribution


@jax.jit
def ringed_lightcurve(state: dict) -> jax.Array:
    """The main function for computing the transit light curve of a ringed planet.

    The ringed analog of :func:`polynomial_limb_darkened_transit.lightcurve`: the same
    vectorized setup and sequential per-timestep ``jax.lax.scan``, but each in-transit
    step evaluates the five-term inclusion-exclusion decomposition described in the
    module docstring instead of a single ellipse-star term. Two properties keep the
    common cases cheap:

    * If the planet's projected outline never pokes out of the inner ring edge (no
      planet-inner-edge crossings, the usual case for a mostly face-on ring), both
      triple terms equal F(P intersect S) exactly and cancel, so only the three
      two-region terms are evaluated.
    * An exactly edge-on ring projects to a degenerate (zero-area) ellipse whose
      counterclockwise traversal doubles back on itself, making every ring term
      integrate to zero automatically -- no special-casing is needed, and the ringed
      light curve collapses to the planet-only one continuously.

    Requires the full 3D parameterization (``parameterize_with_projected_ellipse`` is
    not meaningful for rings, which need the true orbital-frame orientation).

    Args:
        state (dict):
            A dictionary containing all of the keys that are included in a
            :func:`RingedSystem` ``state`` attribute.

    Returns:
        Array:
            The flux received from the star at each time step for the times included
            as ``state["times"]``.

    """
    setup = _lightcurve_setup(state, parameterize_with_projected_ellipse=False)
    g_coeffs = setup["g_coeffs"]
    normalization_constant = setup["normalization_constant"]
    positions = setup["positions"]

    # _lightcurve_setup stored the true anomalies in state["f"], which ring_prelude
    # needs for the ring-edge sky projections
    para_outer, para_inner = ring_prelude(state)

    # the outer ring edge's projected semi-major axis is always exactly ring_outer_r,
    # so it bounds the whole system's projection
    possibly_in_transit = (
        positions[0, :] ** 2 + positions[1, :] ** 2
        <= (1.0 + state["ring_outer_r"] * 1.1) ** 2
    ) * (positions[2, :] > 0)

    def _sentinel_crossings(_: None) -> tuple:
        s = jnp.full(4, 999.0)
        return s, s, s

    def transiting(X: tuple) -> jax.Array:
        p_para, p_two, o_para, i_para = X

        # F(P & S): the planet's implicit conic is always well-conditioned, so the
        # generic term (which solves its quartic from the implicit form) applies
        ps_term = ellipse_star_term(p_para, p_two, g_coeffs)

        # ring-edge crossings, solved with the (possibly nearly-degenerate) ring on
        # the parametric side; each pairwise solve is shared across all terms.
        # A conservative bounding pre-test skips the quartic for ring edges that
        # provably don't straddle the stellar limb (the containment logic in the arc
        # splitter resolves those steps from the sentinel-only crossing sets)
        rs_o = jax.lax.cond(
            ellipse_bound(**o_para),
            lambda _: parametric_conic_intersections(**o_para, **STAR_TWO),
            _sentinel_crossings,
            None,
        )
        rs_i = jax.lax.cond(
            ellipse_bound(**i_para),
            lambda _: parametric_conic_intersections(**i_para, **STAR_TWO),
            _sentinel_crossings,
            None,
        )
        os_term = ring_star_term(o_para, *rs_o, g_coeffs)
        is_term = ring_star_term(i_para, *rs_i, g_coeffs)

        # planet-inner-edge crossings, also skippable: if the planet's bounding circle
        # fits inside the inner edge's inscribed circle (radius det / semi-major, with
        # the semi-major bounded above by the coefficient norm), the concentric curves
        # can't cross
        planet_bound = jnp.sqrt(
            p_para["c_x1"] ** 2
            + p_para["c_x2"] ** 2
            + p_para["c_y1"] ** 2
            + p_para["c_y2"] ** 2
        )
        det_i = i_para["c_x1"] * i_para["c_y2"] - i_para["c_x2"] * i_para["c_y1"]
        inner_axis_ub = jnp.sqrt(
            i_para["c_x1"] ** 2
            + i_para["c_x2"] ** 2
            + i_para["c_y1"] ** 2
            + i_para["c_y2"] ** 2
        )
        pr_i = jax.lax.cond(
            planet_bound * inner_axis_ub < det_i,
            _sentinel_crossings,
            lambda _: parametric_conic_intersections(**i_para, **p_two),
            None,
        )

        def planet_inside_inner_edge(_: None) -> jax.Array:
            # no planet-inner-edge crossings and concentric curves mean P is inside I
            # (and so inside O): both triple terms equal F(P & S) and cancel
            return ps_term + os_term - is_term

        def planet_pokes_through(_: None) -> jax.Array:
            ps_xs, ps_ys = _single_intersection_points(**p_two)
            pr_o = parametric_conic_intersections(**o_para, **p_two)
            triple_o = triple_term(p_para, o_para, ps_xs, ps_ys, *rs_o, *pr_o, g_coeffs)
            triple_i = triple_term(p_para, i_para, ps_xs, ps_ys, *rs_i, *pr_i, g_coeffs)
            return ps_term + os_term - is_term - triple_o + triple_i

        return (
            jax.lax.cond(
                jnp.sum(pr_i[1] == 999) == 4,
                planet_inside_inner_edge,
                planet_pokes_through,
                None,
            )
            * normalization_constant
        )

    def not_transiting(X: tuple) -> float:
        return 0.0

    def scan_func(carry: None, scan_over: tuple) -> tuple[None, jax.Array]:
        p_para, p_two, o_para, i_para, mask = scan_over
        return None, jax.lax.cond(
            mask, transiting, not_transiting, (p_para, p_two, o_para, i_para)
        )

    transit_fluxes = jax.lax.scan(
        scan_func,
        None,
        (setup["para"], setup["two"], para_outer, para_inner, possibly_in_transit),
        None,
    )[1]

    return setup["fluxes"] - transit_fluxes
