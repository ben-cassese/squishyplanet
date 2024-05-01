import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from squishyplanet import OblateSystem

import matplotlib.pyplot as plt


def test_limb_darkening_profile_helper():
    key = jax.random.PRNGKey(0)
    for i in range(1_000):
        key, *subkeys = jax.random.split(key, (3,))
        poly_order = jax.random.randint(subkeys[0], (1,), minval=2, maxval=7)
        poly_order = int(poly_order[0])
        u_coeffs = jax.random.uniform(key, (poly_order,), minval=0.0, maxval=1.0)

        state = {
            "t_peri": 0.0,
            "times": jnp.linspace(0.0, 5, 10),
            "a": 2.0,
            "period": 10,
            "r": 0.1,
            "ld_u_coeffs": u_coeffs,
        }

        system = OblateSystem(**state)

        n = 100_000
        z = system.limb_darkening_profile(jnp.linspace(0, 1, n))
        z *= jnp.linspace(0, 1, n)

        # the ui changed for jnp in #20524, don't want to limit versions just for
        # this though
        try:
            val = jnp.trapezoid(z, jnp.linspace(0, 1, n)) * 2 * jnp.pi
        except:
            val = jnp.trapz(z, jnp.linspace(0, 1, n)) * 2 * jnp.pi

        assert jnp.allclose(val, 1.0)


def test_illustrations():
    state = {
        "t_peri": 0.0,
        "times": jnp.linspace(0.0, 5, 400),
        "a": 2.0,
        "period": 10,
        "r": 0.5,
        "i": 0.0,
        "f1": 0.1,
        "f2": 0.5,
        "obliq": -jnp.pi / 2,
        "tidally_locked": False,
        "ld_u_coeffs": jnp.array([0.99, 0.65]),
    }
    system = OblateSystem(**state)

    system.illustrate(
        times=0.0,
        true_anomalies=None,
        orbit=True,
        reflected=False,
        emitted=False,
        star_fill=True,
        window_size=0.4,
        star_centered=False,
        nsamples=10_000,
        figsize=(8, 8),
    )
    plt.close()

    system.illustrate(
        times=jnp.array([0, 0.5]),
        true_anomalies=None,
        orbit=True,
        reflected=False,
        emitted=False,
        star_fill=True,
        window_size=0.4,
        star_centered=False,
        nsamples=10_000,
        figsize=(8, 8),
    )
    plt.close()

    system.illustrate(
        times=None,
        true_anomalies=0.0,
        orbit=True,
        reflected=False,
        emitted=False,
        star_fill=True,
        window_size=0.4,
        star_centered=False,
        nsamples=10_000,
        figsize=(8, 8),
    )
    plt.close()

    system.illustrate(
        times=None,
        true_anomalies=jnp.array([0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]),
        orbit=True,
        reflected=False,
        emitted=False,
        star_fill=True,
        window_size=0.4,
        star_centered=False,
        nsamples=10_000,
        figsize=(8, 8),
    )
    plt.close()

    system.illustrate(
        times=None,
        true_anomalies=jnp.array([0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]),
        orbit=True,
        reflected=True,
        emitted=False,
        star_fill=True,
        window_size=0.4,
        star_centered=False,
        nsamples=10_000,
        figsize=(8, 8),
    )
    plt.close()

    system.illustrate(
        times=None,
        true_anomalies=jnp.array([0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]),
        orbit=True,
        reflected=False,
        emitted=True,
        star_fill=True,
        window_size=0.4,
        star_centered=False,
        nsamples=10_000,
        figsize=(8, 8),
    )
    plt.close()

    system.illustrate(
        times=None,
        true_anomalies=jnp.array([0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]),
        orbit=True,
        reflected=False,
        emitted=True,
        star_fill=False,
        window_size=0.4,
        star_centered=True,
        nsamples=10_000,
        figsize=(8, 8),
    )
    plt.close()
