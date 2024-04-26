import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# running into some numerical issues for very narrow ellipses-
# e.g., the terminator for a spherical planet at f=1e-6. the rhos are ~1e14 there,
# and the final value for c_y1 is off. everything is fine at f=1e-5 though, so leaving
# for now
@jax.jit
def poly_to_parametric_helper(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00, **kwargs):
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
    rho_00 -= 1

    # the center of the ellipse
    xc = (rho_xy * rho_y0 - 2 * rho_yy * rho_x0) / (4 * rho_xx * rho_yy - rho_xy**2)
    yc = (rho_xy * rho_x0 - 2 * rho_xx * rho_y0) / (4 * rho_xx * rho_yy - rho_xy**2)

    # the rotation angle
    if rho_xx.shape == ():
        a = jnp.ones((1, 2, 2))
    else:
        a = jnp.ones((rho_xx.shape[0], 2, 2))
    a = a.at[:, 0, 0].set(rho_xx)
    a = a.at[:, 0, 1].set(rho_xy / 2)
    a = a.at[:, 1, 0].set(rho_xy / 2)
    a = a.at[:, 1, 1].set(rho_yy)
    l, b = jax.vmap(jnp.linalg.eigh, in_axes=(0))(a)
    lambda_1 = l[:, 0]
    lambda_2 = l[:, 1]
    cosa = b[:, 0, 0]
    sina = b[:, 0, 1]

    # the radii
    k = -(
        rho_yy * rho_x0**2
        - rho_xy * rho_x0 * rho_y0
        + rho_xx * rho_y0**2
        + rho_xy**2 * rho_00
        - 4 * rho_xx * rho_yy * rho_00
    ) / (rho_xy**2 - 4 * rho_xx * rho_yy)
    r1 = 1 / jnp.sqrt(lambda_1 / k)
    r2 = 1 / jnp.sqrt(lambda_2 / k)
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

    def inner(x, y):
        def loss(alpha):
            x_alpha = c_x1 * jnp.cos(alpha) + c_x2 * jnp.sin(alpha) + c_x3
            y_alpha = c_y1 * jnp.cos(alpha) + c_y2 * jnp.sin(alpha) + c_y3

            return (x - x_alpha) ** 2 + (y - y_alpha) ** 2

        # could have just used autograd, but it's simple enough to do by hand
        def grad_loss(alpha):
            return 2 * (c_x2 * jnp.cos(alpha) - c_x1 * jnp.sin(alpha)) * (
                c_x3 - x + c_x1 * jnp.cos(alpha) + c_x2 * jnp.sin(alpha)
            ) + 2 * (c_y2 * jnp.cos(alpha) - c_y1 * jnp.sin(alpha)) * (
                c_y3 - y + c_y1 * jnp.cos(alpha) + c_y2 * jnp.sin(alpha)
            )

        def scan_func(alpha, _):
            # rarely, but sometimes, it actually reaches loss=0 and grad=0
            # in that case, it throws nans when it tries to divide and everything
            # following that is wrong
            l = loss(alpha)
            g = grad_loss(alpha)

            def not_converged(alpha):
                return alpha - l / g, None

            def converged(alpha):
                return alpha, None

            return jax.lax.cond(jnp.abs(g) > 1e-16, not_converged, converged, alpha)

        # if it starts on the actual value, will get nans
        # previously had it start at 0, but in idealized cases used for testing it really is 0
        return jax.lax.scan(scan_func, 0.123, None, length=50)[0]

    alphas = jax.vmap(inner)(xs, ys)
    alphas = jnp.mod(alphas, 2 * jnp.pi)
    alphas = jnp.where(alphas < 0, alphas + 2 * jnp.pi, alphas)
    return alphas
