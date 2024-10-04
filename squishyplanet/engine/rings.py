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

    return {
        "c_x1": cx1,
        "c_x2": cx2,
        "c_x3": cx3,
        "c_y1": cy1,
        "c_y2": cy2,
        "c_y3": cy3,
    }


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


def planet_ring_overlap(precomputed_lc_setup, ring_para, planet_two):
    """
    Compute the flux blocked by a region that's doubly blocked by a planet and an
    ellipse representing the inner or outer edge of a ring.

    This is *not* the full correction factor, just a tool used when computing it

    # nope this doesn't work
    # Logic flow. Honestly easier to read this function from bottom to top.
    # For each timestep:
    # - Is there a chance you're transiting?
    #     - No:   dont_check_for_overlap, return 0.0
    #     - Yes:  Do the planet and ring have any intersection points,
    #             meaning there's potentially a double-counted region?
    #             check_for_overlap
    #             - No:   no_overlap, return 0.0
    #             - Yes:  is any portion of that double-counted region in transit?
    #                     overlap_potentially_transiting
    #                     - No:   overlap_not_transiting, return 0.0
    #                     - Yes:  how many points of intersection are inside the star?
    #                             overlap_transiting
    #                             - 3 or 4:  three_or_four_points_interior
    #                                 - 3: three_points_interior, return integral
    #                                 - 4: four_points_interior, return integral
    #                             - 1 or 2:  one_or_two_points_interior
    #                                 - 1: one_point_interior, return integral
    #                                 - 2: two_points_interior, return integral




    """
    (
        fluxes,
        normalization_constant,
        g_coeffs,
        two,
        para,
        possibly_in_transit,
        positions,
        true_anomalies,
    ) = precomputed_lc_setup

    # def one_or_two_points_interior(args):
    #     intersection_pts, indv_ring_para, indv_planet_two = args
    #     pts_in_star = jnp.sum(intersection_pts[0] ** 2 + intersection_pts[1] ** 2 < 1.0)

    #     def one_point_interior():
    #         pass

    #     def two_points_interior():
    #         pass

    #     return jax.lax.cond(
    #         pts_in_star == 1,
    #         one_point_interior,
    #         two_points_interior,
    #         (intersection_pts, indv_ring_para, indv_planet_two),
    #     )

    # def three_or_four_points_interior(args):
    #     intersection_pts, indv_ring_para, indv_planet_two = args
    #     pts_in_star = jnp.sum(intersection_pts[0] ** 2 + intersection_pts[1] ** 2 < 1.0)

    #     def three_points_interior():
    #         pass

    #     def four_points_interior():
    #         pass

    #     return jax.lax.cond(
    #         pts_in_star == 4,
    #         four_points_interior,
    #         three_points_interior,
    #         (intersection_pts, indv_ring_para, indv_planet_two),
    #     )

    # def overlap_transiting(args):
    #     intersection_pts, indv_ring_para, indv_planet_two = args
    #     pts_in_star = jnp.sum(intersection_pts[0] ** 2 + intersection_pts[1] ** 2 < 1.0)
    #     return jax.lax.cond(
    #         pts_in_star >= 3,
    #         three_or_four_points_interior,
    #         one_or_two_points_interior,
    #         (intersection_pts, indv_ring_para, indv_planet_two),
    #     )

    # def overlap_not_transiting(_):
    #     return 0.0

    # # here the overlap could be not transiting, on the limb, or fully in transit
    # # this routes the timestep to the correct case
    # def overlap_potentially_transiting(args):
    #     indv_ring_para, indv_planet_two = args
    #     intersection_pts = ring_planet_intersection(
    #         **indv_ring_para, **sindv_planet_two
    #     )
    #     pts_in_star = jnp.sum(intersection_pts[0] ** 2 + intersection_pts[1] ** 2 < 1.0)
    #     return jax.lax.cond(
    #         pts_in_star > 0,
    #         overlap_transiting,
    #         overlap_not_transiting,
    #         (intersection_pts, indv_ring_para, indv_planet_two),
    #     )

    # # I realize this is silly to have a repeat of the same function, but it's
    # # easier to keep track in my head this way
    # def no_overlap(_):
    #     return 0.0

    # def check_for_overlap(args):
    #     indv_ring_para, indv_planet_two = args
    #     intersection_pts = ring_planet_intersection(**indv_ring_para, **indv_planet_two)
    #     there_is_overlap = jnp.sum(intersection_pts[0] == 999) < 4
    #     return jax.lax.cond(
    #         there_is_overlap,
    #         overlap_potentially_transiting,
    #         no_overlap,
    #         (indv_ring_para, indv_planet_two),
    #     )

    # def dont_check_for_overlap(_):
    #     return 0.0

    # def scan_func(carry, scan_over):
    #     indv_ring_para, indv_planet_two, mask = scan_over

    #     return None, jax.lax.cond(
    #         mask,
    #         check_for_overlap,
    #         dont_check_for_overlap,
    #         (indv_ring_para, indv_planet_two),
    #     )

    # double_counted_flux = jax.lax.scan()


@jax.jit
def ring_lightcurve(state):

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

    # compute the lightcurves of the three ellipses separately
    ring_outer_lc = lightcurve(
        state=state, parameterize_with_projected_ellipse=False, precomputed=setup
    )

    ring_inner_lc = lightcurve(
        state=state,
        parameterize_with_projected_ellipse=False,
        precomputed={
            "fluxes": fluxes,
            "normalization_constant": normalization_constant,
            "g_coeffs": g_coeffs,
            "two": two_ring_inner,
            "para": para_ring_inner,
            "possibly_in_transit": possibly_in_transit,
            "positions": positions,
            "true_anomalies": true_anomalies,
        },
    )

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

    planet_ring_inner_overlap = 0.0
    planet_ring_outer_overlap = 0.0

    ring_inner_blockage = ring_inner_lc - planet_ring_inner_overlap
    ring_outer_blockage = ring_outer_lc - planet_ring_outer_overlap
    ring_blockage = ring_outer_blockage - ring_inner_blockage

    return planet_lc + ring_blockage
