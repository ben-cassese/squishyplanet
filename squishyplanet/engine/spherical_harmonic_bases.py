import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import sph_harm as jsph_harm

from squishyplanet.engine.spherical_harmonics import parse_n


@jax.jit
def _sph(l, m, theta, phi):
    """
    Compute the real spherical harmonics of degree l and order m at specific point(s)

    Only actually used in visualizations, all fluxes are computed using the 1D integral
    form.

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

    # man this took awhile to track down: starry *also* throws in a factor of
    # (2/np.sqrt(np.pi)) to the normalization. Not stated explicitly in the
    # paper, but must fall out of the normalization to make the "average"
    # flux 1.0
    return sol.real * (
        2 / jnp.sqrt(jnp.pi)
    )  # the imaginary part should be within ~machine precision of 0 anyway


@jax.jit
def sph(x, y, coeffs):
    z = jnp.sqrt(1 - x**2 - y**2)
    phis = jnp.arctan2(y, x)
    thetas = jnp.arccos(z)

    def additions(coeff, i):
        l, m = parse_n(i)
        c = (
            jax.vmap(_sph)(
                jnp.ones_like(phis).astype(int) * (l),
                jnp.ones_like(phis).astype(int) * (m),
                thetas,
                phis,
            )
            * coeffs[i]
        )
        return c

    c = jnp.sum(jax.vmap(additions)(coeffs, jnp.arange(coeffs.shape[0])), axis=0)

    return c


@jax.jit
def poly(x, y, coeffs, a1):
    p = jnp.matmul(a1, coeffs)
    z = jnp.sqrt(1 - x**2 - y**2)

    # eq 8
    l, m = parse_n(jnp.arange(coeffs.shape[0]))
    mu = l - m
    nu = l + m

    # eq 7
    def inner(mu, nu):
        p_tilde = jnp.where(
            (nu % 2 == 0),
            x ** (mu / 2) * y ** (nu / 2),
            x ** ((mu - 1) / 2) * y ** ((nu - 1) / 2) * z,
        )
        return p_tilde

    return jnp.sum(jax.vmap(inner)(mu, nu) * p[:, None, None], axis=0) * (
        2 / jnp.sqrt(jnp.pi)
    )


@jax.jit
def poly_squish(f, x, y, coeffs, a1):
    p = jnp.matmul(a1, coeffs)
    z = (1 - f) * jnp.sqrt(1 - x**2 - y**2)

    # eq 8
    l, m = parse_n(jnp.arange(coeffs.shape[0]))
    mu = l - m
    nu = l + m

    # eq 7
    def inner(mu, nu):
        p_tilde = jnp.where(
            (nu % 2 == 0),
            x ** (mu / 2) * y ** (nu / 2),
            x ** ((mu - 1) / 2) * y ** ((nu - 1) / 2) * z,
        )
        return p_tilde

    return jnp.sum(jax.vmap(inner)(mu, nu) * p[:, None, None], axis=0) * (
        2 / jnp.sqrt(jnp.pi)
    )
