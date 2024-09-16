import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def poly_to_parametric_helper(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    """
    A helper function for :func:`poly_to_parametric`.

    Args:
        rho_xx (Array [Dimensionless]): Coefficient of x^2
        rho_xy (Array [Dimensionless]): Coefficient of xy
        rho_x0 (Array [Dimensionless]): Coefficient of x
        rho_yy (Array [Dimensionless]): Coefficient of y^2
        rho_y0 (Array [Dimensionless]): Coefficient of y
        rho_00 (Array [Dimensionless]): Constant term

    Returns:
        Tuple:
            - r1 (Array [Rstar]): Semi-major axis of the projected ellipse
            - r2 (Array [Rstar]): Semi-minor axis of the projected ellipse
            - xc (Array [Rstar]): x-coordinate of the center of the ellipse
            - yc (Array [Rstar]): y-coordinate of the center of the ellipse
            - cosa (Array [Dimensionless]): Cosine of the rotation angle
            - sina (Array [Dimensionless]): Sine of the rotation angle
    """
    # (* base eq *)
    # pxx  x^2 + pxy  x  y + px0 x + pyy y^2 + py0 y + p00 == 1
    # (* normalize to get rid of p0 *)
    # pxx/(1 - p00)  x^2 + pxy /(1 - p00) x  y + px0/(1 - p00) x +
    # pyy/(1 - p00) y^2 + py0 /(1 - p00) y == 1
    # (* solve for the ellipse center *)
    # CoefficientRules[
    # pxx/(1 - p00)  x^2 + pxy /(1 - p00) x  y + px0/(1 - p00) x +
    # pyy/(1 - p00) y^2 + py0 /(1 - p00) y /. {x -> x - xc,
    # y -> y - yc}, {x, y}]
    # Solve[{px0/(1 - p00) - (2 pxx xc)/(1 - p00) - (pxy yc)/(1 - p00) == 0,
    # py0/(1 - p00) - (pxy xc)/(1 - p00) - (2 pyy yc)/(1 - p00) ==
    # 0 }, {xc, yc}]
    # (* plug back in *)
    # Simplify[
    # CoefficientRules[
    # pxx/(1 - p00)  x^2 + pxy /(1 - p00) x  y + px0/(1 - p00) x +
    #     pyy/(1 - p00) y^2 + py0 /(1 - p00) y /. {x -> x - xc,
    #     y -> y - yc} /. {xc -> -((-pxy py0 + 2 px0 pyy)/(
    #     pxy^2 - 4 pxx pyy)),
    #     yc -> -((-px0 pxy + 2 pxx py0)/(pxy^2 - 4 pxx pyy))}, {x, y}]]
    # (* normalize again to get the final coeffs *)
    # pxxShift =
    # Simplify[(pxx/(
    #     1 - p00)) /(1 - (
    #     px0 pxy py0 - pxx py0^2 -
    #     px0^2 pyy)/((-1 + p00) (pxy^2 - 4 pxx pyy)))]
    # pxyShift =
    # Simplify[(pxy/(
    #     1 - p00))/(1 - (
    #     px0 pxy py0 - pxx py0^2 -
    #     px0^2 pyy)/((-1 + p00) (pxy^2 - 4 pxx pyy)))]
    # pyyShift =
    # Simplify[(pyy/(
    #     1 - p00)) /(1 - (
    #     px0 pxy py0 - pxx py0^2 -
    #     px0^2 pyy)/((-1 + p00) (pxy^2 - 4 pxx pyy)))]

    # the center of the ellipse
    xc = (rho_xy * rho_y0 - 2 * rho_yy * rho_x0) / (4 * rho_xx * rho_yy - rho_xy**2)
    yc = (rho_xy * rho_x0 - 2 * rho_xx * rho_y0) / (4 * rho_xx * rho_yy - rho_xy**2)

    # get new coefficients for the centered ellipse: all others are zero now,
    # explicitly got rid of rho_00 so there's a lot more divison
    rho_xx_shift = -(
        (rho_xx * (rho_xy**2 - 4 * rho_xx * rho_yy))
        / (
            (-1 + rho_00) * rho_xy**2
            - rho_x0 * rho_xy * rho_y0
            + rho_x0**2 * rho_yy
            + rho_xx * (rho_y0**2 + 4 * rho_yy - 4 * rho_00 * rho_yy)
        )
    )
    rho_xy_shift = (-(rho_xy**3) + 4 * rho_xx * rho_xy * rho_yy) / (
        (-1 + rho_00) * rho_xy**2
        - rho_x0 * rho_xy * rho_y0
        + rho_x0**2 * rho_yy
        + rho_xx * (rho_y0**2 + 4 * rho_yy - 4 * rho_00 * rho_yy)
    )
    rho_yy_shift = -(
        (rho_yy * (rho_xy**2 - 4 * rho_xx * rho_yy))
        / (
            (-1 + rho_00) * rho_xy**2
            - rho_x0 * rho_xy * rho_y0
            + rho_x0**2 * rho_yy
            + rho_xx * (rho_y0**2 + 4 * rho_yy - 4 * rho_00 * rho_yy)
        )
    )

    # get the rotation angle (edge case gives you nans if there's no rotation)
    theta = jnp.where(
        rho_xx_shift - rho_yy_shift != 0.0,
        0.5 * jnp.arctan2(rho_xy_shift, (rho_xx_shift - rho_yy_shift)) + jnp.pi / 2,
        0.0,
    )
    theta = jnp.where(theta < 0.0, theta + jnp.pi, theta)
    # jax.debug.print("{x}", x=theta)
    cosa = jnp.cos(theta)
    sina = jnp.sin(theta)

    # get the semi-major and semi-minor axes
    a = (
        rho_xx_shift * jnp.cos(theta) ** 2
        + rho_xy_shift * jnp.cos(theta) * jnp.sin(theta)
        + rho_yy_shift * jnp.sin(theta) ** 2
    )
    b = (
        rho_xx_shift * jnp.sin(theta) ** 2
        - rho_xy_shift * jnp.cos(theta) * jnp.sin(theta)
        + rho_yy_shift * jnp.cos(theta) ** 2
    )

    r1 = 1 / jnp.sqrt(a)
    r2 = 1 / jnp.sqrt(b)

    return r1, r2, xc, yc, cosa, sina


@jax.jit
def poly_to_parametric(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    """
    Convert between the coefficients that describe an implicit to those
    defining a parametric ellipse.

    The input coefficients are those of the implicit ellipse equation:

    .. math::
        \\rho_{xx} x^2 + \\rho_{xy} xy + \\rho_{x0} x + \\rho_{yy} y^2 + \\rho_{y0} y + \\rho_{00} = 1



    Args:
        rho_xx (Array [Dimensionless]): Coefficient of x^2
        rho_xy (Array [Dimensionless]): Coefficient of xy
        rho_x0 (Array [Dimensionless]): Coefficient of x
        rho_yy (Array [Dimensionless]): Coefficient of y^2
        rho_y0 (Array [Dimensionless]): Coefficient of y
        rho_00 (Array [Dimensionless]): Constant term

    Returns:
        dict:
            Dictionary of coefficients for the parametric ellipse. The ellipse can now
            be described by the following parametric equations for parameter :math:`\\alpha`:

            .. math::
                x = c_{x1} * \\cos(\\alpha) + c_{x2} * \\sin(\\alpha) + c_{x3}
                y = c_{y1} * \\cos(\\alpha) + c_{y2} * \\sin(\\alpha) + c_{y3}
    """
    r1, r2, xc, yc, cosa, sina = poly_to_parametric_helper(
        rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00
    )

    return {
        "c_x1": r1 * cosa,
        "c_x2": -r2 * sina,
        "c_x3": xc,
        "c_y1": r1 * sina,
        "c_y2": r2 * cosa,
        "c_y3": yc,
    }


@jax.jit
def cartesian_intersection_to_parametric_angle(
    xs,
    ys,
    c_x1,
    c_x2,
    c_x3,
    c_y1,
    c_y2,
    c_y3,
):
    """
    Given a set of x and y coordinates corresponding to the intersection of the planet
    and star, compute the angle :math:`\\alpha` that corresponds to each point.

    Here, :math:`\\alpha` is the parameter in the parametric equations of the ellipse.
    See :func:`poly_to_parametric` for more details.

    Args:
        xs (Array [Rstar]): x-coordinates of the intersection points
        ys (Array [Rstar]): y-coordinates of the intersection points
        c_x1 (Array [Dimensionless]): Coefficient of x^2
        c_x2 (Array [Dimensionless]): Coefficient of xy
        c_x3 (Array [Dimensionless]): Coefficient of x
        c_y1 (Array [Dimensionless]): Coefficient of y^2
        c_y2 (Array [Dimensionless]): Coefficient of y
        c_y3 (Array [Dimensionless]): Constant term

    Returns:
        Array [Rstar]: The angle :math:`\\alpha` corresponding to each intersection point

    """

    # center the ellipse
    xs -= c_x3
    ys -= c_y3

    # the x, y positions are now linear combinations of just cosa, sina
    # linear solve for those
    inv = jnp.linalg.inv(jnp.array([[c_x1, c_x2], [c_y1, c_y2]]))
    matrix = jax.vmap(lambda x, y: jnp.matmul(inv, jnp.array([x, y])))(xs, ys)
    cosa = matrix[:, 0]
    sina = matrix[:, 1]

    # convert to alpha
    alpha = jnp.arctan2(sina, cosa)
    return alpha
