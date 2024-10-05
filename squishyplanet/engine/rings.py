import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from squishyplanet.engine.planet_3d import planet_3d_coeffs
from squishyplanet.engine.planet_2d import planet_2d_coeffs
from squishyplanet.engine.parametric_ellipse import (
    poly_to_parametric,
    parametric_to_poly,
    cartesian_intersection_to_parametric_angle,
)
from squishyplanet.engine.polynomial_limb_darkened_transit import (
    _lightcurve_setup,
    lightcurve,
)


@jax.jit
def ring_para_coeffs(a, e, f, Omega, i, omega, rRing, ring_obliq, ring_prec, **kwargs):
    """
    Compute the coefficients describing the parametric form of a ring in the sky plane.

    Remember that ring_obliq and ring_prec are defined *in the planet's orbital plane, at f=0*.
    That implies that to get a face-on ring, you need ring_obliq=ring_prec=90 deg. Some other
    examples all at inc=90 deg:
    - ring_obliq=0, ring_prec=0: the ring is an edge-on horizontal line
    - ring_obliq=jnp.pi/4, ring_prec=0: the ring is still a line, but now it's tilted 45 degrees in the sky frame. You've tipped the north pole away from the star.
    - ring_obliq=0, ring_prec=anything: the ring is still a line
    - ring_obliq=90, ring_prec=0: the ring is a face-on circle
    Making inc != 90 deg will alter these: the angles are defined in the orbital plane,
    so if you tilt the orbit away from face-on, you'll also tilt the ring away from face-on.

    Args:
        a (Array): The semi-major axis of the planet.
        e (Array): The eccentricity of the planet.
        f (Array): The true anomaly of the planet.
        Omega (Array): The longitude of the ascending node of the planet.
        i (Array): The inclination of the planet.
        omega (Array): The argument of periapsis of the planet.
        rRing (Array): The radius of the ring.
        ring_obliq (Array): The ring_obliquity of the ring.
        ring_prec (Array): The ring_precession of the ring.
        kwargs (dict): Additional (unused) keyword arguments.

    Returns:
        dict:
            A dictionary with keys for each of the coefficients of the parametric
            form of the ring.
    """
    cx1 = rRing * (
        -(jnp.sin(i) * jnp.sin(ring_obliq) * jnp.sin(Omega))
        - jnp.cos(ring_obliq)
        * jnp.sin(ring_prec)
        * (
            jnp.cos(Omega) * jnp.sin(omega)
            + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
        )
        + jnp.cos(ring_prec)
        * jnp.cos(ring_obliq)
        * (
            jnp.cos(omega) * jnp.cos(Omega)
            - jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
        )
    )
    cx2 = -(
        rRing
        * (
            jnp.cos(omega)
            * (
                jnp.cos(Omega) * jnp.sin(ring_prec)
                + jnp.cos(i) * jnp.cos(ring_prec) * jnp.sin(Omega)
            )
            + jnp.sin(omega)
            * (
                jnp.cos(ring_prec) * jnp.cos(Omega)
                - jnp.cos(i) * jnp.sin(ring_prec) * jnp.sin(Omega)
            )
        )
    )
    cx3 = (
        a
        * (-1 + e**2)
        * (
            jnp.sin(f)
            * (
                jnp.cos(Omega) * jnp.sin(omega)
                + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
            )
            + jnp.cos(f)
            * (
                -(jnp.cos(omega) * jnp.cos(Omega))
                + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
            )
        )
    ) / (1 + e * jnp.cos(f))

    cy1 = rRing * (
        jnp.cos(Omega)
        * (
            jnp.sin(i) * jnp.sin(ring_obliq)
            + jnp.cos(i) * jnp.cos(ring_obliq) * jnp.sin(ring_prec + omega)
        )
        + jnp.cos(ring_obliq) * jnp.cos(ring_prec + omega) * jnp.sin(Omega)
    )
    cy2 = rRing * (
        jnp.cos(i) * jnp.cos(ring_prec + omega) * jnp.cos(Omega)
        - jnp.sin(ring_prec + omega) * jnp.sin(Omega)
    )
    cy3 = -(
        (
            a
            * (-1 + e**2)
            * (
                jnp.cos(i) * jnp.cos(Omega) * jnp.sin(f + omega)
                + jnp.cos(f + omega) * jnp.sin(Omega)
            )
        )
        / (1 + e * jnp.cos(f))
    )

    coeffs = {
        "c_x1": cx1,
        "c_x2": cx2,
        "c_x3": cx3,
        "c_y1": cy1,
        "c_y2": cy2,
        "c_y3": cy3,
    }

    if coeffs["c_x1"].shape != coeffs["c_x3"].shape:
        coeffs["c_x1"] = jnp.ones_like(coeffs["c_x3"]) * coeffs["c_x1"]
        coeffs["c_x2"] = jnp.ones_like(coeffs["c_x3"]) * coeffs["c_x2"]
        coeffs["c_y1"] = jnp.ones_like(coeffs["c_x3"]) * coeffs["c_y1"]
        coeffs["c_y2"] = jnp.ones_like(coeffs["c_x3"]) * coeffs["c_y2"]

    return coeffs


@jax.jit
def ring_planet_intersection(
    c_x1, c_x2, c_x3, c_y1, c_y2, c_y3, rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00
):
    """
    The cartesian intersection points in the sky-planet between the projected planet and a ring edge.

    You need to provide the coefficients that describe the ring and the planet in the
    sky plane, one of which is in parametric form, and the other in implicit form,
    though it doesn't matter which is which.

    This is a more general version of `_single_intersection_point` in
    `polynomial_limb_darkened_transit`, which assume the second "ellipse" is the star.

    The rho coefficients satisfy the following equation for either the ring or the planet:

    .. math::
                \\rho__{xx} x^2 + \\rho__{xy} xy + \\rho__{x0} x + \\rho__{yy} y^2 + \\rho__{y0} y + \\rho__{00} = 1

    And the c coefficients describe the parametric curve of either the planet or the ring:

    .. math::
                x = c_x1 \\cos(t) + c_x2 \\sin(t) + c_x3
                y = c_y1 \\cos(t) + c_y2 \\sin(t) + c_y3

    Args:
        c_x1 (Array): Coefficient in the parametric form of the planet/ring.
        c_x2 (Array): Coefficient in the parametric form of the planet/ring.
        c_x3 (Array): Coefficient in the parametric form of the planet/ring.
        c_y1 (Array): Coefficient in the parametric form of the planet/ring.
        c_y2 (Array): Coefficient in the parametric form of the planet/ring.
        c_y3 (Array): Coefficient in the parametric form of the planet/ring.
        rho_xx (Array): Coefficient in the implicit form of the ring/planet.
        rho_xy (Array): Coefficient in the implicit form of the ring/planet.
        rho_x0 (Array): Coefficient in the implicit form of the ring/planet.
        rho_yy (Array): Coefficient in the implicit form of the ring/planet.
        rho_y0 (Array): Coefficient in the implicit form of the ring/planet.
        rho_00 (Array): Coefficient in the implicit form of the ring/planet.

    Returns:
        Tuple:
            A tuple of two arrays, the first array contains the x-coordinates of the
            intersection points, and the second array contains the y-coordinates of the
            intersection points. Imaginary roots of the quartic polynomial that
            correspond to no intersection are replaced with 999.

    """

    t4 = (
        -1
        + rho_00
        + c_x1**2 * rho_xx
        + c_x3**2 * rho_xx
        + c_x3 * (rho_x0 + (-c_y1 + c_y3) * rho_xy)
        - c_x1 * (rho_x0 + 2 * c_x3 * rho_xx + (-c_y1 + c_y3) * rho_xy)
        - c_y1 * rho_y0
        + c_y3 * rho_y0
        + (c_y1 - c_y3) ** 2 * rho_yy
    )

    t3 = 2 * (
        c_x2
        * (
            rho_x0
            - 2 * c_x1 * rho_xx
            + 2 * c_x3 * rho_xx
            - c_y1 * rho_xy
            + c_y3 * rho_xy
        )
        + c_y2
        * (
            -(c_x1 * rho_xy)
            + c_x3 * rho_xy
            + rho_y0
            - 2 * c_y1 * rho_yy
            + 2 * c_y3 * rho_yy
        )
    )

    t2 = 2 * (
        -1
        + rho_00
        - c_x1**2 * rho_xx
        + c_x3**2 * rho_xx
        - c_x1 * c_y1 * rho_xy
        + 2 * c_x2 * (c_x2 * rho_xx + c_y2 * rho_xy)
        + c_x3 * (rho_x0 + c_y3 * rho_xy)
        + c_y3 * rho_y0
        + (-(c_y1**2) + 2 * c_y2**2 + c_y3**2) * rho_yy
    )

    t1 = 2 * (
        c_x2 * (rho_x0 + 2 * (c_x1 + c_x3) * rho_xx + (c_y1 + c_y3) * rho_xy)
        + c_y2 * ((c_x1 + c_x3) * rho_xy + rho_y0 + 2 * (c_y1 + c_y3) * rho_yy)
    )

    t0 = (
        -1
        + rho_00
        + c_x1**2 * rho_xx
        + c_x3**2 * rho_xx
        + c_x3 * (rho_x0 + (c_y1 + c_y3) * rho_xy)
        + c_x1 * (rho_x0 + 2 * c_x3 * rho_xx + (c_y1 + c_y3) * rho_xy)
        + (c_y1 + c_y3) * (rho_y0 + (c_y1 + c_y3) * rho_yy)
    )

    polys = jnp.array([t4, t3, t2, t1, t0])
    roots = jnp.roots(polys, strip_zeros=False)  # strip_zeros must be False to jit

    ts = jnp.where(jnp.imag(roots) == 0, jnp.real(roots), 999)
    cos_t = (1 - ts**2) / (1 + ts**2)
    sin_t = 2 * ts / (1 + ts**2)
    xs = jnp.where(ts != 999, c_x1 * cos_t + c_x2 * sin_t + c_x3, ts)
    ys = jnp.where(ts != 999, c_y1 * cos_t + c_y2 * sin_t + c_y3, ts)

    return xs, ys


def transit_of_region_with_overlap():
    pass


def combo_lightcurve(
    para_ring,
    two_ring,
    para_planet,
    two_planet,
    these_times_have_intersections,
    pts,
    precomputed,
):
    jax.debug.print("beginning combo_lightcurve")

    # if the overlapping region is not transiting, we can just use the old lightcurve
    # function for the ring and the planet individually. This usually means only one of
    # the two are transiting, but you can construct a weird scenario where both are and
    # you have two closed curves
    # jax.debug.print("pts radial dist: {x}", x=pts[0]**2 + pts[1]**2)
    # jax.debug.print("inside star? {x}", x=jnp.sum(pts[0]**2 + pts[1]**2 < 1, axis=1))
    at_least_one_intersection_inside_star = (
        jnp.sum(pts[0] ** 2 + pts[1] ** 2 < 1, axis=1) > 0
    )
    # jax.debug.print("at_least_one_intersection_inside_star: {x}", x=at_least_one_intersection_inside_star)

    # first, figure out where the overlapping region is not transiting.
    # at these times, get the lcs individually. One will probably be zero.
    mask = precomputed["possibly_in_transit"]
    # jax.debug.print("mask: {x}", x=mask)
    mask *= these_times_have_intersections
    # jax.debug.print("mask: {x}", x=mask)
    mask *= ~at_least_one_intersection_inside_star
    # jax.debug.print("mask: {x}", x=mask)

    # jax.debug.print("possibly_in_transit: {x}", x=precomputed["possibly_in_transit"])
    # jax.debug.print("these_times_have_intersections: {x}", x=these_times_have_intersections)
    # jax.debug.print("at_least_one_intersection_inside_star: {x}", x=at_least_one_intersection_inside_star)
    # jax.debug.print("mask: {x}", x=mask)
    jax.debug.print(
        "mask of times when could be transiting, there are intersections, but not in the star, so indv lcs: {x}",
        x=mask,
    )
    lone_planet_lc = (
        lightcurve(
            {"_": 0.0},  # need to provide something, but it's not used
            False,  # same
            {
                "fluxes": precomputed["fluxes"],
                "normalization_constant": precomputed["normalization_constant"],
                "g_coeffs": precomputed["g_coeffs"],
                "two": two_planet,
                "para": para_planet,
                "possibly_in_transit": mask,
                "positions": precomputed["positions"],
                "true_anomalies": precomputed["true_anomalies"],
            },
        )
        - 1.0
    )
    lone_ring_lc = (
        lightcurve(
            {"_": 0.0},  # need to provide something, but it's not used
            False,  # same
            {
                "fluxes": precomputed["fluxes"],
                "normalization_constant": precomputed["normalization_constant"],
                "g_coeffs": precomputed["g_coeffs"],
                "two": two_ring,
                "para": para_ring,
                "possibly_in_transit": mask,
                "positions": precomputed["positions"],
                "true_anomalies": precomputed["true_anomalies"],
            },
        )
        - 1.0
    )
    lone_curve_contributions = lone_planet_lc + lone_ring_lc
    jax.debug.print("lone_curve_contributions: {x}", x=lone_curve_contributions)

    # simultaneous_transit_contributions = ouch()


def ring_lightcurve(para_ring, two_ring, para_planet, two_planet, precomputed):
    """
    Compute the light curve created by one of the ring boundaries and the planet.

    If the two ellipses don't intesect, since the ring is guaranteed to be larger that
    the planet, the light curve is just that of the ring boundary and we can use the old
    lightcurve function. If they do intersect though, we might need to trace the outline
    of the combination of the two ellipses. That's done in combo_lightcurve.
    """

    def are_there_intersections(indv_para_ring, indv_two_planet):
        pts = ring_planet_intersection(**indv_para_ring, **indv_two_planet)
        return jnp.sum(pts[0] != 999) > 0, pts

    # this will be all or none of the points in the time series, unless the projected
    # shape of the planet+ring are evolving during the transit
    these_times_have_intersections, pts = jax.vmap(are_there_intersections)(
        para_ring, two_planet
    )
    # return these_times_have_intersections, pts

    # we're going to trick the old lightcurve function into only computing the fluxes
    # for the times that don't have intersections (again, usually will be all or
    # nothing) by modifying the possibly_in_transit mask to also block out times
    # that have intersections
    vanilla_lc_mask = ~these_times_have_intersections
    vanillia_lc_mask = vanilla_lc_mask * precomputed["possibly_in_transit"]
    jax.debug.print(
        "mask of times when could be transiting, but no intersections, so indv ring-only lc: {x}",
        x=vanilla_lc_mask,
    )
    vanilla_lc_precomputed = {
        "fluxes": precomputed["fluxes"],
        "normalization_constant": precomputed["normalization_constant"],
        "g_coeffs": precomputed["g_coeffs"],
        "two": two_ring,
        "para": para_ring,
        "possibly_in_transit": vanilla_lc_mask,
        "positions": precomputed["positions"],
        "true_anomalies": precomputed["true_anomalies"],
    }

    # temporarily commenting out to speed testing of combo_lightcurve
    no_interesction_lc = (
        lightcurve(
            {"_": 0.0},  # need to provide something, but it's not used
            False,  # same
            vanilla_lc_precomputed,
        )
        - 1.0
    )
    jax.debug.print("no_interesctions_lc: {x}", x=no_interesction_lc)

    intersections_lc = combo_lightcurve(
        para_ring,
        two_ring,
        para_planet,
        two_planet,
        these_times_have_intersections,
        pts,
        precomputed,
    )
    jax.debug.print("intersections_lc: {x}", x=intersections_lc)

    return no_interesction_lc + intersections_lc


@jax.jit
def ringed_system_lightcurve(state):

    # compute things that can be shared across all three lc integrations
    # switch the "r" to the outer ring radius, which is guaranteed to be largest,
    # when making the mask if it's potentially in transit
    _state = state.copy()
    _state["r"] = state["ring_outer_r"]
    _state["obliq"] = state["ring_obliq"]
    _state["prec"] = state["ring_prec"]
    setup = _lightcurve_setup(_state, parameterize_with_projected_ellipse=False)
    normalization_constant = setup["normalization_constant"]
    g_coeffs = setup["g_coeffs"]
    possibly_in_transit = setup["possibly_in_transit"]
    positions = setup["positions"]
    true_anomalies = setup["true_anomalies"]

    # these two are specific to the outer ring
    two_ring_outer = setup["two"]
    para_ring_outer = setup["para"]

    # get the coeffs of the inner ring
    para_ring_inner = ring_para_coeffs(**state, rRing=state["ring_inner_r"])
    two_ring_inner = poly_to_parametric(**para_ring_inner)

    # get the coeffs of the planet
    three_planet = planet_3d_coeffs(**state)
    two_planet = planet_2d_coeffs(**three_planet)

    # get the flux blocked by the planet
    planet_lc = lightcurve(
        state=state,
        parameterize_with_projected_ellipse=False,
        precomputed={
            "fluxes": fluxes,
            "normalization_constant": normalization_constant,
            "g_coeffs": g_coeffs,
            "two": two_planet,
            "para": three_planet,
            "possibly_in_transit": possibly_in_transit,
            "positions": positions,
            "true_anomalies": true_anomalies,
        },
    )

    outer_ring_lc = ring_lightcurve(
        para_ring_inner, two_ring_inner, para_planet, two_planet, setup
    )
    inner_ring_lc = combo_transit(
        para_ring_inner, two_ring_inner, para_planet, two_planet, setup
    )

    ring_lc = outer_ring_lc - inner_ring_lc
    return planet_lc + ring_lc
