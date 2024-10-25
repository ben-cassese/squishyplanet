import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from squishyplanet.limb_darkening_laws import (
    linear_ld_law,
    quadratic_ld_law,
    kipping_ld_law,
    squareroot_ld_law,
    nonlinear_3param_ld_law,
    nonlinear_4param_ld_law,
)


def test_limb_darkening_laws():
    mus = jnp.linspace(0, 1, 100)

    coeffs = linear_ld_law(u1=0.5)
    assert jnp.allclose(coeffs, jnp.array([0.5, 0.0]))
    profile = linear_ld_law(u1=0.5, return_profile=True)
    diff = (
        profile["true_intensity_profile"] - profile["squishyplanet_intensity_profile"]
    )
    assert jnp.max(jnp.abs(diff)) < 0.1

    coeffs = quadratic_ld_law(u1=0.5, u2=0.2)
    assert jnp.allclose(coeffs, jnp.array([0.5, 0.2]))
    profile = quadratic_ld_law(u1=0.5, u2=0.2, return_profile=True)
    diff = (
        profile["true_intensity_profile"] - profile["squishyplanet_intensity_profile"]
    )
    assert jnp.max(jnp.abs(diff)) < 0.1

    coeffs = kipping_ld_law(q1=0.5, q2=0.2)
    assert jnp.allclose(coeffs, jnp.array([0.28284271, 0.42426407]))
    profile = kipping_ld_law(q1=0.5, q2=0.2, return_profile=True)
    diff = (
        profile["true_intensity_profile"] - profile["squishyplanet_intensity_profile"]
    )
    assert jnp.max(jnp.abs(diff)) < 0.1

    coeffs = squareroot_ld_law(u1=0.5, u2=0.2)
    profile = squareroot_ld_law(u1=0.5, u2=0.2, return_profile=True)
    diff = (
        profile["true_intensity_profile"] - profile["squishyplanet_intensity_profile"]
    )
    assert jnp.max(jnp.abs(diff)) < 0.1

    coeffs = nonlinear_3param_ld_law(u1=0.5, u2=0.2, u3=0.1)
    profile = nonlinear_3param_ld_law(u1=0.5, u2=0.2, u3=0.1, return_profile=True)
    diff = (
        profile["true_intensity_profile"] - profile["squishyplanet_intensity_profile"]
    )
    assert jnp.max(jnp.abs(diff)) < 0.1

    coeffs = nonlinear_4param_ld_law(u1=0.5, u2=0.2, u3=0.1, u4=0.05)
    profile = nonlinear_4param_ld_law(
        u1=0.5, u2=0.2, u3=0.1, u4=0.05, return_profile=True
    )
    diff = (
        profile["true_intensity_profile"] - profile["squishyplanet_intensity_profile"]
    )
    assert jnp.max(jnp.abs(diff)) < 0.1
