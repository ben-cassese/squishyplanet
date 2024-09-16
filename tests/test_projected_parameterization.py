import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from squishyplanet import OblateSystem

import astropy.units as u


def test_projected_parameterization():
    key = jax.random.PRNGKey(0)
    t_exp = 5 * u.min
    times = jnp.arange(-30, 30, t_exp.to(u.hour).value) * u.hour.to(u.day)

    for i in range(2_000):
        key, *subkeys = jax.random.split(key, (5,))

        # the 3d planet
        injected_state = {
            "t_peri": -1071.23205 / 4,
            "times": times,
            "exposure_time": t_exp.to(u.day).value,
            "oversample": 5,  # 5x more samples under-the-hood, then binned back down
            "oversample_correction_order": 2,
            "a": 540.0,
            "period": 1071.23205,
            "r": 0.1,
            "i": 89.9720 * jnp.pi / 180,
            "ld_u_coeffs": jnp.array([0.4, 0.26]),
            "f1": jax.random.uniform(subkeys[0], (1,), minval=0.0, maxval=0.99),
            "f2": jax.random.uniform(subkeys[1], (1,), minval=0.0, maxval=0.99),
            "obliq": jax.random.uniform(subkeys[2], (1,), minval=0.0, maxval=jnp.pi),
            "prec": jax.random.uniform(subkeys[3], (1,), minval=0.0, maxval=jnp.pi),
            "tidally_locked": False,
        }

        planet1 = OblateSystem(**injected_state)
        lc1 = planet1.lightcurve()

        # the 2d planet
        injected_state2 = {
            "t_peri": -1071.23205 / 4,
            "times": times,
            "exposure_time": t_exp.to(u.day).value,
            "oversample": 5,  # 5x more samples under-the-hood, then binned back down
            "oversample_correction_order": 2,
            "a": 540.0,
            "period": 1071.23205,
            "i": 89.9720 * jnp.pi / 180,
            "ld_u_coeffs": jnp.array([0.4, 0.26]),
            "tidally_locked": False,
            "parameterize_with_projected_ellipse": True,
            "projected_effective_r": planet1.state["projected_effective_r"][0],
            "projected_f": planet1.state["projected_f"],
            "projected_theta": planet1.state["projected_theta"],
        }

        planet2 = OblateSystem(**injected_state2)
        lc2 = planet2.lightcurve()

        # looking out for spikes
        assert jnp.all(lc1 <= 1.0)
        assert jnp.all(lc2 <= 1.0)

        r = planet1.state["projected_effective_r"][0]
        assert jnp.min(lc1 >= r**2)
        assert jnp.min(lc2 >= r**2)

        # single point outliers
        diff = jnp.diff(lc1)
        assert jnp.abs(jnp.argmax(diff) - jnp.argmin(diff)) != 1
        diff = jnp.diff(lc2)
        assert jnp.abs(jnp.argmax(diff) - jnp.argmin(diff)) != 1

        assert jnp.allclose(lc1, lc2)
