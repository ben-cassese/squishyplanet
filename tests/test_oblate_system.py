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
        u_coeffs = jax.random.uniform(subkeys[1], (poly_order,), minval=0.0, maxval=1.0)

        n = 100_000
        z = OblateSystem.limb_darkening_profile(
            ld_u_coeffs=u_coeffs, r=jnp.linspace(0, 1, n)
        )
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


def test_lightcurve():
    state = {
        "times": jnp.linspace(-1, 1, 10),
        "tidally_locked": False,
        "compute_reflected_phase_curve": False,
        "compute_emitted_phase_curve": False,
        "compute_stellar_ellipsoidal_variations": False,
        "compute_stellar_doppler_variations": False,
        "parameterize_with_projected_ellipse": True,
        "projected_effective_r": 0.1,
        "projected_f": 0.1,
        "projected_theta": 0.1,
        "phase_curve_nsamples": 10,
        "random_seed": 0,
        "exposure_time": 0.0,
        "oversample": 1,
        "oversample_correction_order": 2,
        "e": 0.0,
        "Omega": jnp.pi,
        "omega": 0.0,
        "obliq": 0.0,
        "prec": 0.0,
        "hotspot_latitude": -jnp.pi / 2,
        "t_peri": 0.5,
        "period": 1.0,
        "a": 4.0,
        "i": jnp.pi,
        "r": 0.1,
        "f1": 0.0,
        "f2": 0.0,
        "ld_u_coeffs": jnp.array([0.1, 0.04]),
        "hotspot_longitude": 0.0,
        "hotspot_concentration": 0.2,
        "albedo": 1.0,
        "emitted_scale": 1e-5,
        "stellar_ellipsoidal_alpha": 1e-6,
        "stellar_doppler_alpha": 1e-6,
        "systematic_trend_coeffs": jnp.array([0.0, 0.0]),
        "log_jitter": -10.0,
        "data": jnp.array([1.0]),
        "uncertainties": jnp.array([0.01]),
    }
    planet = OblateSystem(**state)
    _ = planet.lightcurve()

    state["parameterize_with_projected_ellipse"] = False
    state["tidally_locked"] = True
    state["compute_reflected_phase_curve"] = True
    state["compute_emitted_phase_curve"] = True
    state["compute_stellar_ellipsoidal_variations"] = True
    state["compute_stellar_doppler_variations"] = True

    planet = OblateSystem(**state)
    _ = planet.lightcurve()

    # for key, value in state.items():
    #     if (type(value) != int) & (type(value) != bool):
    #         if key in ["times", "data", "uncertainties", "exoposure_time"]:
    #             continue
    #         new_val = value + 0.01
    #         _ = planet.lightcurve({key: new_val})

    # # now toggle the other settings
    # state["parameterize_with_projected_ellipse"] = False
    # planet = OblateSystem(**state)
    # for key, value in state.items():
    #     if (type(value) != int) & (type(value) != bool):
    #         if key in ["times", "data", "uncertainties", "exoposure_time"]:
    #             continue
    #         new_val = value + 0.01
    #         _ = planet.lightcurve({key: new_val})

    # state["tidally_locked"] = True
    # planet = OblateSystem(**state)
    # for key, value in state.items():
    #     if (type(value) != int) & (type(value) != bool):
    #         if key in ["times", "data", "uncertainties", "exoposure_time"]:
    #             continue
    #         new_val = value + 0.01
    #         _ = planet.lightcurve({key: new_val})

    # # do the reflected and emitted separately, then together
    # state["compute_reflected_phase_curve"] = True
    # planet = OblateSystem(**state)
    # for key, value in state.items():
    #     if (type(value) != int) & (type(value) != bool):
    #         if key in ["times", "data", "uncertainties", "exoposure_time"]:
    #             continue
    #         new_val = value + 0.01
    #         _ = planet.lightcurve({key: new_val})

    # state["compute_reflected_phase_curve"] = False
    # state["compute_emitted_phase_curve"] = True
    # planet = OblateSystem(**state)
    # for key, value in state.items():
    #     if (type(value) != int) & (type(value) != bool):
    #         if key in ["times", "data", "uncertainties", "exoposure_time"]:
    #             continue
    #         new_val = value + 0.01
    #         _ = planet.lightcurve({key: new_val})

    # state["compute_reflected_phase_curve"] = True
    # state["compute_emitted_phase_curve"] = True
    # planet = OblateSystem(**state)
    # for key, value in state.items():
    #     if (type(value) != int) & (type(value) != bool):
    #         if key in ["times", "data", "uncertainties", "exoposure_time"]:
    #             continue
    #         new_val = value + 0.01
    #         _ = planet.lightcurve({key: new_val})

    # # now the stellar variations
    # state["compute_reflected_phase_curve"] = False
    # state["compute_emitted_phase_curve"] = False
    # state["compute_stellar_ellipsoidal_variations"] = True
    # state["compute_stellar_doppler_variations"] = True
    # planet = OblateSystem(**state)
    # for key, value in state.items():
    #     if (type(value) != int) & (type(value) != bool):
    #         if key in ["times", "data", "uncertainties", "exoposure_time"]:
    #             continue
    #         new_val = value + 0.01
    #         _ = planet.lightcurve({key: new_val})

    # # everyone together
    # state["compute_reflected_phase_curve"] = True
    # state["compute_emitted_phase_curve"] = True
    # state["compute_stellar_ellipsoidal_variations"] = True
    # state["compute_stellar_doppler_variations"] = True
    # planet = OblateSystem(**state)
    # for key, value in state.items():
    #     if (type(value) != int) & (type(value) != bool):
    #         if key in ["times", "data", "uncertainties", "exoposure_time"]:
    #             continue
    #         new_val = value + 0.01
    #         _ = planet.lightcurve({key: new_val})


def test_loglike():
    state = {
        "times": jnp.linspace(-1, 1, 100),
        "data": jnp.ones(100),
        "uncertainties": jnp.ones(100) * 0.01,
        "tidally_locked": True,
        "compute_reflected_phase_curve": True,
        "compute_emitted_phase_curve": True,
        "compute_stellar_ellipsoidal_variations": True,
        "compute_stellar_doppler_variations": True,
        "parameterize_with_projected_ellipse": False,
        "projected_effective_r": 0.0,
        "projected_f": 0.0,
        "projected_theta": 0.0,
        "phase_curve_nsamples": 100,
        "random_seed": 0,
        "exposure_time": 0.0,
        "oversample": 1,
        "oversample_correction_order": 2,
        "e": 0.0,
        "Omega": jnp.pi,
        "omega": 0.0,
        "obliq": 0.0,
        "prec": 0.0,
        "hotspot_latitude": -jnp.pi / 2,
        "t_peri": 0.5,
        "period": 1.0,
        "a": 4.0,
        "i": jnp.pi,
        "r": 0.1,
        "f1": 0.0,
        "f2": 0.0,
        "ld_u_coeffs": jnp.array([0.1, 0.04]),
        "hotspot_longitude": 0.0,
        "hotspot_concentration": 0.2,
        "albedo": 1.0,
        "emitted_scale": 1e-5,
        "stellar_ellipsoidal_alpha": 1e-6,
        "stellar_doppler_alpha": 1e-6,
        "systematic_trend_coeffs": jnp.array([0.0, 0.0]),
        "log_jitter": -10.0,
    }
    planet = OblateSystem(**state)
    _ = planet.loglike()
