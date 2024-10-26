# These are implementations heavily based on those in  ExoTIC-LD
# (ads:2024JOSS....9.6816G). Many thanks to the authors for their work!
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial
from squishyplanet import OblateSystem

mu_grid = jnp.linspace(0, 1, 500)


@partial(jax.jit, static_argnames=("return_profile"))
def linear_ld_law(u1, return_profile=False):
    """
    Linear limb-darkening law.

    .. math::
        \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu)

    Args:
        u1 (float): Linear limb-darkening coefficient
        return_profile (bool, default=False):
            Whether to return a dictionary describing the intensity profile

    Returns:
        Array:
            u coefficients used by squishyplanet to compute the intensity profile
            (here it's just [u1, 0], since we always need at least two coefficients)
        dict (if return_profile=True):
            Dictionary describing the intensity profile
    """
    u_coeffs = jnp.array([u1, 0])

    if return_profile:
        # squishyplanet normalizes intensity profiles so that their integrated flux is 1
        # but when plotting we need to renormalize to where mu=0
        norm = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=jnp.array([1.0])
        )
        poly_model = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=mu_grid
        )
        return {
            "mu_grid": mu_grid,
            "true_intensity_profile": 1 - u1 * (1 - mu_grid),
            "squishyplanet_intensity_profile": poly_model / norm,
            "u_coeffs": u_coeffs,
        }

    return u_coeffs


@partial(jax.jit, static_argnames=("return_profile"))
def quadratic_ld_law(u1, u2, return_profile=False):
    """
    Quadratic limb-darkening law.

    .. math::
        \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu) - u_2 (1 - \\mu)^2

    Args:
        u1 (float): Linear limb-darkening coefficient
        u2 (float): Quadratic limb-darkening coefficient
        return_profile (bool, default=False):
            Whether to return a dictionary describing the intensity profile

    Returns:
        Array:
            u coefficients used by squishyplanet to compute the intensity profile
            (here it's just [u1, u2], without modification: this is a silly function
            included only to give a consistent interface to the limb-darkening laws)
        dict (if return_profile=True):
            Dictionary describing the intensity profile
    """
    u_coeffs = jnp.array([u1, u2])

    if return_profile:
        # squishyplanet normalizes intensity profiles so that their integrated flux is 1
        # but when plotting we need to renormalize to where mu=0
        norm = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=jnp.array([1.0])
        )
        poly_model = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=mu_grid
        )
        return {
            "mu_grid": mu_grid,
            "true_intensity_profile": 1 - u1 * (1 - mu_grid) - u2 * (1 - mu_grid) ** 2,
            "squishyplanet_intensity_profile": poly_model / norm,
            "u_coeffs": u_coeffs,
        }

    return u_coeffs


@partial(jax.jit, static_argnames=("return_profile"))
def kipping_ld_law(q1, q2, return_profile=False):
    """
    Kipping limb-darkening law.

    A restriction of the quadratic law from Kipping 2013 that guaratees a monotonic
    increasing intensity profile

    .. math::
        \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu) - u_2 (1 - \\mu)^2

    where

    .. math::
        u_1 = 2 \\sqrt{q_1} q_2

    .. math::
        u_2 = \\sqrt{q_1} (1 - 2 q_2)

    Args:
        q1 (float):
            Kipping limb-darkening coefficient
        q2 (float):
            Kipping limb-darkening coefficient
        return_profile (bool, default=False):
            Whether to return a dictionary describing the intensity profile

    Returns:
        Array or dict:
            If `return_profile` is False, returns an array of `u` coefficients used by
            squishyplanet to compute the intensity profile.

            If `return_profile` is True, returns a dictionary describing the intensity
            profile.
    """
    u1 = 2.0 * q1**0.5 * q2
    u2 = q1**0.5 * (1 - 2.0 * q2)
    u_coeffs = jnp.array([u1, u2])

    if return_profile:
        # squishyplanet normalizes intensity profiles so that their integrated flux is 1
        # but when plotting we need to renormalize to where mu=0
        norm = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=jnp.array([1.0])
        )
        poly_model = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=mu_grid
        )
        return {
            "mu_grid": mu_grid,
            "true_intensity_profile": 1 - u1 * (1 - mu_grid) - u2 * (1 - mu_grid) ** 2,
            "squishyplanet_intensity_profile": poly_model / norm,
            "u_coeffs": u_coeffs,
        }

    return u_coeffs


@partial(jax.jit, static_argnames=("return_profile"))
def squareroot_ld_law(u1, u2, order=12, return_profile=False):
    """
    Square root limb-darkening law.

    .. math::
        \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu) - u_2 (1 - \\sqrt{\\mu})

    Args:
        u1 (float): Linear limb-darkening coefficient
        u2 (float): Square root limb-darkening coefficient
        order (int, default=5): Order of the polynomial fit to the intensity profile
        return_profile (bool, default=False):
            Whether to return a dictionary describing the intensity profile

    Returns:
        Array:
            u coefficients of the least-squares polynomial fit to the intensity profile
            created by the limb-darkening law across a dense grid of mu values
        dict (if return_profile=True):
            Dictionary describing the intensity profile
    """
    intensity_profile = 1 - u1 * (1 - mu_grid) - u2 * (1 - mu_grid**0.5)
    u_coeffs = OblateSystem.fit_limb_darkening_profile(
        intensities=intensity_profile, mus=mu_grid, order=order
    )

    if return_profile:
        # squishyplanet normalizes intensity profiles so that their integrated flux is 1
        # but when plotting we need to renormalize to where mu=0
        norm = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=jnp.array([1.0])
        )
        poly_model = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=mu_grid
        )
        return {
            "mu_grid": mu_grid,
            "true_intensity_profile": intensity_profile,
            "squishyplanet_intensity_profile": poly_model / norm,
            "u_coeffs": u_coeffs,
        }

    return u_coeffs


@partial(jax.jit, static_argnames=("return_profile"))
def nonlinear_3param_ld_law(u1, u2, u3, order=12, return_profile=False):
    """
    Non-linear 3-parameter limb-darkening law.

    .. math::
        \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu) - u_2 (1 - \\mu^{1.5}) - u_3 (1 - \\mu^2)

    Args:
        u1 (float): Linear limb-darkening coefficient
        u2 (float): Square root limb-darkening coefficient
        u3 (float): Square limb-darkening coefficient
        order (int, default=5): Order of the polynomial fit to the intensity profile
        return_profile (bool, default=False):
            Whether to return a dictionary describing the intensity profile

    Returns:
        Array:
            u coefficients of the least-squares polynomial fit to the intensity profile
            created by the limb-darkening law across a dense grid of mu values
        dict (if return_profile=True):
            Dictionary describing the intensity profile
    """
    intensity_profile = (
        1 - u1 * (1 - mu_grid) - u2 * (1 - mu_grid**1.5) - u3 * (1 - mu_grid**2)
    )
    u_coeffs = OblateSystem.fit_limb_darkening_profile(
        intensities=intensity_profile, mus=mu_grid, order=order
    )

    if return_profile:
        # squishyplanet normalizes intensity profiles so that their integrated flux is 1
        # but when plotting we need to renormalize to where mu=0
        norm = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=jnp.array([1.0])
        )
        poly_model = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=mu_grid
        )
        return {
            "mu_grid": mu_grid,
            "true_intensity_profile": intensity_profile,
            "squishyplanet_intensity_profile": poly_model / norm,
            "u_coeffs": u_coeffs,
        }

    return u_coeffs


@partial(jax.jit, static_argnames=(["order", "return_profile"]))
def nonlinear_4param_ld_law(u1, u2, u3, u4, order=12, return_profile=False):
    """
    Non-linear 4-parameter limb-darkening law.

    .. math::
        \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu^{0.5}) - u_2 (1 - \\mu) - u_2 (1 - \\mu^{1.5}) - u_3 (1 - \\mu^2)

    Args:
        u1 (float): Linear limb-darkening coefficient
        u2 (float): Square root limb-darkening coefficient
        u3 (float): Square limb-darkening coefficient
        u4 (float): Square limb-darkening coefficient
        order (int, default=5): Order of the polynomial fit to the intensity profile
        return_profile (bool, default=False):
            Whether to return a dictionary describing the intensity profile

    Returns:
        Array:
            u coefficients of the least-squares polynomial fit to the intensity profile
            created by the limb-darkening law across a dense grid of mu values
        dict (if return_profile=True):
            Dictionary describing the intensity profile
    """
    intensity_profile = (
        1
        - u1 * (1 - mu_grid**0.5)
        - u2 * (1 - mu_grid)
        - u3 * (1 - mu_grid**1.5)
        - u4 * (1 - mu_grid**2)
    )
    u_coeffs = OblateSystem.fit_limb_darkening_profile(
        intensities=intensity_profile, mus=mu_grid, order=order
    )

    if return_profile:
        # squishyplanet normalizes intensity profiles so that their integrated flux is 1
        # but when plotting we need to renormalize to where mu=0
        norm = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=jnp.array([1.0])
        )
        poly_model = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, mu=mu_grid
        )
        return {
            "mu_grid": mu_grid,
            "true_intensity_profile": intensity_profile,
            "squishyplanet_intensity_profile": poly_model / norm,
            "u_coeffs": u_coeffs,
        }

    return u_coeffs
