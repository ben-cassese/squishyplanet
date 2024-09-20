import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def ring_poly_coeffs_2d(r, obliq, prec, xc, yc):
    """
    Compute the 2D implicit coefficients describing a circle after transformation into the sky plane.

    Args:
        r (Array): The radius of the circle.
        obliq (Array): The obliquity of the circle.
        prec (Array): The precession of the circle.
        xc (Array): The sky-projected x-coordinate of the circle's center.
        yc (Array): The sky-projected y-coordinate of the circle's center.

    Returns:
        dict:
            A dictionary with keys representing different transformed coefficient
            names ('rho_xx', 'rho_xy', 'rho_x0', 'rho_yy', 'rho_y0', 'rho_00') and
            their corresponding values. These coefficients describe the outline of the
            planet as an implicit curve that satisfies the equation:

            .. math::
                \\rho_{xx} x^2 + \\rho_{xy} xy + \\rho_{x0} x + \\rho_{yy} y^2 + \\rho_{y0} y + \\rho_{00} = 1
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
