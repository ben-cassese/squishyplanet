import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def ring_poly_coeffs_2d(r, obliq, prec, xc, yc):
    """
    Compute the 2D implicit coefficients describing a circle after transformation into the sky plane.

    Originally from Tiger Lu, written for Lu et al. 2024,
    "HIP-41378 f Likely has High Obliquity – Implications for Exorings"

    Args:
        r (Array): The radius of the circle.
        obliq (Array): The obliquity of the circle.
        prec (Array): The precession of the circle.
        xc (Array): The sky-projected x-coordinate of the circle's center.
        yc (Array): The sky-projected y-coordinate of the circle's center.

    Returns:
        dict:
            A dictionary with keys representing different transformed coefficient
            names ('rho__xx', 'rho__xy', 'rho__x0', 'rho__yy', 'rho__y0', 'rho__00') and
            their corresponding values. These coefficients describe the outline of the
            planet as an implicit curve that satisfies the equation:

            .. math::
                \\rho__{xx} x^2 + \\rho__{xy} xy + \\rho__{x0} x + \\rho__{yy} y^2 + \\rho__{y0} y + \\rho__{00} = 1
    """

    q_x1 = jnp.sin(obliq) ** 2
    q_x2 = 2 * jnp.cos(obliq) * jnp.sin(obliq) * jnp.sin(prec)
    q_x3 = -2 * xc * jnp.sin(obliq) ** 2 - 2 * yc * jnp.cos(obliq) * jnp.sin(
        obliq
    ) * jnp.sin(prec)
    q_y1 = jnp.cos(prec) ** 2 + jnp.cos(obliq) ** 2 * jnp.sin(prec) ** 2
    q_y2 = (
        -2 * yc * jnp.cos(prec) ** 2
        - 2 * xc * jnp.cos(obliq) * jnp.sin(obliq) * jnp.sin(prec)
        - 2 * yc * jnp.cos(obliq) ** 2 * jnp.sin(prec) ** 2
    )
    q_y3 = 0

    norm = (
        r**2 * jnp.sin(obliq) ** 2 * jnp.cos(prec) ** 2
        - yc**2 * jnp.cos(prec) ** 2
        - xc**2 * jnp.sin(obliq) ** 2
        - 2 * xc * yc * jnp.cos(obliq) * jnp.sin(obliq) * jnp.sin(prec)
        - yc**2 * jnp.cos(obliq) ** 2 * jnp.sin(prec) ** 2
    )

    return {
        "rho_xx": q_x1 / norm,
        "rho_xy": q_x2 / norm,
        "rho_x0": q_x3 / norm,
        "rho_yy": q_y1 / norm,
        "rho_y0": q_y2 / norm,
        "rho_00": q_y3 / norm,
    }


def _t4(
    c_x1, c_x2, c_x3, c_y1, c_y2, c_y3, rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00
):
    return (
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


def _t3(
    c_x1, c_x2, c_x3, c_y1, c_y2, c_y3, rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00
):
    return 2 * (
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


def _t2(
    c_x1, c_x2, c_x3, c_y1, c_y2, c_y3, rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00
):
    return 2 * (
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


def _t1(
    c_x1, c_x2, c_x3, c_y1, c_y2, c_y3, rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00
):
    return 2 * (
        c_x2 * (rho_x0 + 2 * (c_x1 + c_x3) * rho_xx + (c_y1 + c_y3) * rho_xy)
        + c_y2 * ((c_x1 + c_x3) * rho_xy + rho_y0 + 2 * (c_y1 + c_y3) * rho_yy)
    )


def _t0(
    c_x1, c_x2, c_x3, c_y1, c_y2, c_y3, rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00
):
    return (
        -1
        + rho_00
        + c_x1**2 * rho_xx
        + c_x3**2 * rho_xx
        + c_x3 * (rho_x0 + (c_y1 + c_y3) * rho_xy)
        + c_x1 * (rho_x0 + 2 * c_x3 * rho_xx + (c_y1 + c_y3) * rho_xy)
        + (c_y1 + c_y3) * (rho_y0 + (c_y1 + c_y3) * rho_yy)
    )


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

    t4 = _t4(
        c_x1,
        c_x2,
        c_x3,
        c_y1,
        c_y2,
        c_y3,
        rho_xx,
        rho_xy,
        rho_x0,
        rho_yy,
        rho_y0,
        rho_00,
    )
    t3 = _t3(
        c_x1,
        c_x2,
        c_x3,
        c_y1,
        c_y2,
        c_y3,
        rho_xx,
        rho_xy,
        rho_x0,
        rho_yy,
        rho_y0,
        rho_00,
    )
    t2 = _t2(
        c_x1,
        c_x2,
        c_x3,
        c_y1,
        c_y2,
        c_y3,
        rho_xx,
        rho_xy,
        rho_x0,
        rho_yy,
        rho_y0,
        rho_00,
    )
    t1 = _t1(
        c_x1,
        c_x2,
        c_x3,
        c_y1,
        c_y2,
        c_y3,
        rho_xx,
        rho_xy,
        rho_x0,
        rho_yy,
        rho_y0,
        rho_00,
    )
    t0 = _t0(
        c_x1,
        c_x2,
        c_x3,
        c_y1,
        c_y2,
        c_y3,
        rho_xx,
        rho_xy,
        rho_x0,
        rho_yy,
        rho_y0,
        rho_00,
    )

    polys = jnp.array([t4, t3, t2, t1, t0])
    roots = jnp.roots(polys, strip_zeros=False)  # strip_zeros must be False to jit

    ts = jnp.where(jnp.imag(roots) == 0, jnp.real(roots), 999)
    cos_t = (1 - ts**2) / (1 + ts**2)
    sin_t = 2 * ts / (1 + ts**2)
    xs = jnp.where(ts != 999, c_x1 * cos_t + c_x2 * sin_t + c_x3, ts)
    ys = jnp.where(ts != 999, c_y1 * cos_t + c_y2 * sin_t + c_y3, ts)

    return xs, ys
