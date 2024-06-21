import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import sph_harm as jsph_harm

from functools import partial


@jax.jit
def parse_n(n):
    """
    Convert the nth index in a ylm-coefficient array into the corresponding l, m indices

    Args:
        n (Array, int, shape=(1,)): Index of the ylm-coefficient array.

    Returns:
        Array: l index of the ylm-coefficient array.
        Array: m index of the ylm-coefficient array.
    """
    l = jnp.floor(jnp.sqrt(n))
    m = n - l * l - l
    return l.astype(jnp.int64), m.astype(jnp.int64)


@jax.jit
def sph(l, m, theta, phi):
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
    )  # / norm  # the imaginary part should be within ~machine precision of 0 anyway


@partial(jax.jit, static_argnums=(0))
def _big_G(n, x, y, z):
    """
    Compute G, part of the integrand of the flux integral

    This is Eq. 34 in Luger+ 2019

    Args:
        n int: Index of the ylm-coefficient array.
        x (Array): x coordinate of the point.
        y (Array): y coordinate of the point.
        z (Array): z coordinate of the point.

    Returns:


    """
    # part of the integrand of the flux integral
    # x, y, z in unit sphere, l, m the spherical harmonic
    l, m = parse_n(n)

    # eq 8 in Luger+ 2019
    mu = l - m
    nu = l + m

    # all eq 34 in Luger+ 2019

    # only 3/5 cases x is non-zero
    big_G_x = 0
    big_G_x = jnp.where(
        (l == 1) & (m == 0), (1 - z**3) / (3 * (1 - z**2)) * (-y), big_G_x
    )
    big_G_x = jnp.where(
        ((nu % 2 == 1) & (mu == 1) & (l % 2 == 0)), x ** (l - 2) * z**3, big_G_x
    )
    big_G_x = jnp.where(
        ((nu % 2 == 1) & (mu == 1) & (l % 2 == 1) & (big_G_x == 0)),
        x ** (l - 3) * y * z**3,
        big_G_x,
    )

    # only 3/5 cases y is non-zero
    big_G_y = 0
    big_G_y = jnp.where(nu % 2 == 0, x ** ((mu + 2) / 2) * y ** (nu / 2), big_G_y)
    big_G_y = jnp.where(
        (l == 1) & (m == 0), (1 - z**3) / (3 * (1 - z**2)) * (x), big_G_y
    )
    big_G_y = jnp.where(
        (big_G_x == 0) & (big_G_y == 0),
        x ** ((mu - 3) / 2) * y ** ((nu - 1) / 2) * z**3,
        big_G_y,
    )

    return jnp.stack((big_G_x, big_G_y))


@jax.jit
def big_G(n, x, y, z):
    """
    Compute G, part of the integrand of the flux integral

    This is Eq. 34 in Luger+ 2019

    Args:
        n (Array, shape=(n_max,)):
            An array of indices of the ylm-coefficient array (jnp.arange(n_max)).
        x (Array): x coordinate of the point.
        y (Array): y coordinate of the point.
        z (Array): z coordinate of the point.

    Returns:
        Array: G, part of the integrand of the flux integral.

    """
    return jax.vmap(_big_G, in_axes=(0, None, None, None))(n, x, y, z)
