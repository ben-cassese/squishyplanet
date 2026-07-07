import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def para_eval(alpha: jax.Array, coeffs: dict[str, jax.Array]) -> tuple:
    """Evaluate a parametric ellipse at the given angle(s).

    Args:
        alpha (Array [Radian]): The parametric angle(s) at which to evaluate the curve.
        coeffs (dict): Dictionary of the parametric coefficients c_x1 through c_y3.

    Returns:
        Tuple:
            The x and y coordinates of the curve at each alpha.

    """
    return (
        coeffs["c_x1"] * jnp.cos(alpha)
        + coeffs["c_x2"] * jnp.sin(alpha)
        + coeffs["c_x3"],
        coeffs["c_y1"] * jnp.cos(alpha)
        + coeffs["c_y2"] * jnp.sin(alpha)
        + coeffs["c_y3"],
    )


@jax.jit
def ring_para_coeffs(
    a: jax.Array,
    e: jax.Array,
    f: jax.Array,
    Omega: jax.Array,
    i: jax.Array,
    omega: jax.Array,
    rRing: jax.Array,
    ring_obliq: jax.Array,
    ring_prec: jax.Array,
    **kwargs: jax.Array,
) -> dict[str, jax.Array]:
    """
    Compute the coefficients describing the parametric form of a ring in the sky plane.

    Remember that ring_obliq and ring_prec are defined *in the planet's orbital plane,
    at f=0*. That implies that to get a face-on ring, you need
    ring_obliq=ring_prec=90 deg. Some other examples all at inc=90 deg:

    - ring_obliq=0, ring_prec=0: the ring is an edge-on horizontal line
    - ring_obliq=jnp.pi/4, ring_prec=0: the ring is still a line, but now it's tilted
      45 degrees in the sky frame. You've tipped the north pole away from the star.
    - ring_obliq=0, ring_prec=anything: the ring is still a line
    - ring_obliq=90, ring_prec=0: the ring is a face-on circle

    Making inc != 90 deg will alter these: the angles are defined in the orbital plane,
    so if you tilt the orbit away from face-on, you'll also tilt the ring away from
    face-on.

    The returned coefficients are always oriented so that the curve is traversed
    counterclockwise on the sky as the parametric angle increases, as required for the
    Green's theorem boundary integrals.

    Args:
        a (Array [Rstar]): The semi-major axis of the planet.
        e (Array [Dimensionless]): The eccentricity of the planet.
        f (Array [Radian]): The true anomaly of the planet.
        Omega (Array [Radian]): The longitude of the ascending node of the planet.
        i (Array [Radian]): The inclination of the planet.
        omega (Array [Radian]): The argument of periapsis of the planet.
        rRing (Array [Rstar]): The radius of the ring.
        ring_obliq (Array [Radian]): The obliquity of the ring.
        ring_prec (Array [Radian]): The precession angle of the ring.
        **kwargs: Additional (unused) keyword arguments.

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

    # the raw parameterization's traversal direction flips sign with viewing
    # geometry; enforce counterclockwise (positive determinant) by reversing the
    # direction (alpha -> -alpha, i.e. flipping the sin terms) where needed
    det = cx1 * cy2 - cx2 * cy1
    cx2 = jnp.where(det < 0, -cx2, cx2)
    cy2 = jnp.where(det < 0, -cy2, cy2)

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


def ring_prelude(state: dict) -> tuple[dict, dict]:
    """Per-timestep parametric coefficients for both ring edges.

    Expects ``state["f"]`` to already hold the per-timestep true anomalies (i.e. run
    :func:`polynomial_limb_darkened_transit._lightcurve_setup` first). When the ring
    orientation tracks the planet's (``ring_tracks_planet``), a tidally locked planet
    drags the ring precession angle along with the true anomaly, mirroring the
    ``prec`` override in :func:`polynomial_limb_darkened_transit.outline_prelude`.

    Args:
        state (dict):
            A :func:`RingedSystem` ``state`` dictionary. Uses the orbital elements
            plus ``ring_inner_r``, ``ring_outer_r``, ``ring_obliq``, ``ring_prec``,
            ``ring_tracks_planet``, and ``tidally_locked``.

    Returns:
        Tuple:
            Two dictionaries of parametric coefficients (each broadcast to the time
            axis): the outer ring edge and the inner ring edge.

    """
    # rings that track the planet take their orientation from the planet's obliq/prec
    # at call time, so parameter updates passed to lightcurve() stay consistent
    ring_obliq = jnp.where(
        state["ring_tracks_planet"], state["obliq"], state["ring_obliq"]
    )
    ring_prec = jnp.where(
        state["ring_tracks_planet"], state["prec"], state["ring_prec"]
    )
    ring_prec = jnp.where(
        jnp.logical_and(state["tidally_locked"], state["ring_tracks_planet"]),
        state["f"],
        ring_prec,
    )
    shared = dict(
        a=state["a"],
        e=state["e"],
        f=state["f"],
        Omega=state["Omega"],
        i=state["i"],
        omega=state["omega"],
        ring_obliq=ring_obliq,
        ring_prec=ring_prec,
    )
    para_outer = ring_para_coeffs(rRing=state["ring_outer_r"], **shared)
    para_inner = ring_para_coeffs(rRing=state["ring_inner_r"], **shared)
    return para_outer, para_inner


@jax.jit
def parametric_conic_intersections(
    c_x1: jax.Array,
    c_x2: jax.Array,
    c_x3: jax.Array,
    c_y1: jax.Array,
    c_y2: jax.Array,
    c_y3: jax.Array,
    rho_xx: jax.Array,
    rho_xy: jax.Array,
    rho_x0: jax.Array,
    rho_yy: jax.Array,
    rho_y0: jax.Array,
    rho_00: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Intersections between a parametric ellipse and an implicitly-defined conic.

    This is a more general version of :func:`_single_intersection_points` in
    `polynomial_limb_darkened_transit`, which assumes the implicit curve is the unit
    circle (the star). Substituting the parametric curve into the implicit equation
    via the tangent half-angle substitution :math:`t = \\tan(\\alpha/2)` yields a
    quartic in t.

    The parametric curve should be the one at risk of degeneracy (e.g. a nearly
    edge-on ring): its coefficients stay bounded, whereas the implicit form of a thin
    ellipse has coefficients that diverge as the inverse squared semi-minor axis.

    The implicit coefficients satisfy the following on the conic:

    .. math::
        \\rho_{xx} x^2 + \\rho_{xy} xy + \\rho_{x0} x + \\rho_{yy} y^2 + \\rho_{y0} y + \\rho_{00} = 1

    And the c coefficients describe the parametric curve:

    .. math::
        x = c_{x1} \\cos(\\alpha) + c_{x2} \\sin(\\alpha) + c_{x3}
        y = c_{y1} \\cos(\\alpha) + c_{y2} \\sin(\\alpha) + c_{y3}

    Args:
        c_x1 (Array [Rstar]): Coefficient in the parametric form of the first curve.
        c_x2 (Array [Rstar]): Coefficient in the parametric form of the first curve.
        c_x3 (Array [Rstar]): Coefficient in the parametric form of the first curve.
        c_y1 (Array [Rstar]): Coefficient in the parametric form of the first curve.
        c_y2 (Array [Rstar]): Coefficient in the parametric form of the first curve.
        c_y3 (Array [Rstar]): Coefficient in the parametric form of the first curve.
        rho_xx (Array [Dimensionless]): Coefficient in the implicit form of the second
            curve.
        rho_xy (Array [Dimensionless]): Coefficient in the implicit form of the second
            curve.
        rho_x0 (Array [Dimensionless]): Coefficient in the implicit form of the second
            curve.
        rho_yy (Array [Dimensionless]): Coefficient in the implicit form of the second
            curve.
        rho_y0 (Array [Dimensionless]): Coefficient in the implicit form of the second
            curve.
        rho_00 (Array [Dimensionless]): Coefficient in the implicit form of the second
            curve.

    Returns:
        Tuple:
            Three arrays of length 4: the parametric angles alpha of the intersection
            points on the first curve (in (-pi, pi]), and the x and y coordinates of
            those points. Slots corresponding to imaginary quartic roots (no
            intersection) are filled with 999.

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
    alphas = jnp.where(ts != 999, 2 * jnp.arctan(ts), 999)
    cos_t = (1 - ts**2) / (1 + ts**2)
    sin_t = 2 * ts / (1 + ts**2)
    xs = jnp.where(ts != 999, c_x1 * cos_t + c_x2 * sin_t + c_x3, ts)
    ys = jnp.where(ts != 999, c_y1 * cos_t + c_y2 * sin_t + c_y3, ts)

    return alphas, xs, ys
