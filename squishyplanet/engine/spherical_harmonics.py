import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import sph_harm as jsph_harm

# from jax.scipy.special import lpmv as jlpmn
from functools import partial


@jax.jit
def sph(l, m, theta, phi):
    """
    Compute the real spherical harmonics of degree l and order m at specific point(s)

    Args:
        l (Array, int, shape=(1,)): Degree of the spherical harmonics.
        m (Array, int, shape=(1,)): Order of the spherical harmonics.
        theta (Array, shape=(1,)): Polar angle of the point.
        phi (Array, shape=(1,)): Azimuthal angle of the point.

    Returns:
        Array: Value of the spherical harmonics at the point(s) (theta, phi).

    """
    sol = jnp.zeros_like(theta)
    sol = jnp.where(
        m < 0,
        1j
        / jnp.sqrt(2)
        * (
            jsph_harm(m, l, phi, theta, 25)
            - (-1) ** m * jsph_harm(-m, l, phi, theta, 25)
        ),
        sol,
    )
    sol = jnp.where(
        m > 0,
        1
        / jnp.sqrt(2)
        * (
            jsph_harm(-m, l, phi, theta, 25)
            + (-1) ** m * jsph_harm(m, l, phi, theta, 25)
        ),
        sol,
    )
    sol = jnp.where(m == 0, jsph_harm(0, l, phi, theta, 25), sol)
    return (
        sol.real
    )  # the imaginary part should be within ~machine precision of 0 anyway
