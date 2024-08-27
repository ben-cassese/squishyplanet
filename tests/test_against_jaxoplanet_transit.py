import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxoplanet.orbits.keplerian import Central, Body, System
from jaxoplanet.units import unit_registry as ureg
from jaxoplanet import units
from jaxoplanet.light_curves import limb_dark_light_curve

import matplotlib.pyplot as plt
from tqdm import tqdm

from squishyplanet.engine.polynomial_limb_darkened_transit import lightcurve
from squishyplanet.engine.kepler import kepler
from squishyplanet import OblateSystem

N_JAXOPLANT_COMPARISONS = 1000
TIMES = jnp.linspace(-1, 1, 1728) * ureg.day  # 100s cadence for 48 hours
POLY_ORDERS = [2, 3, 4]


def light_curve_compare(key, poly_limbdark_order, return_lc=False):
    t = TIMES

    key, *rand_key = jax.random.split(key, num=8)

    u = jax.random.uniform(rand_key[6], shape=(poly_limbdark_order,))
    star_mass = jax.random.uniform(rand_key[0], minval=0.1, maxval=1.5) * ureg.M_sun
    semimajor_axis = jax.random.uniform(rand_key[1], minval=0.005, maxval=5.0) * ureg.au
    impact_param = jax.random.uniform(rand_key[2], minval=0.0, maxval=1.0)
    planet_rad = jax.random.uniform(rand_key[3], minval=0.001, maxval=0.25) * ureg.R_sun
    eccentricity = jax.random.uniform(rand_key[4], minval=0.0, maxval=0.9)
    omega = jax.random.uniform(rand_key[5], minval=0.0, maxval=2 * jnp.pi)
    Omega = jnp.pi

    # generate jaxoplanet light curve
    # jaxoplanet works in physical units where the star has mass,
    # can't specify period and semimajor axis
    star = Central(radius=1 * ureg.R_sun, mass=star_mass)
    planet = (
        System(star)
        .add_body(
            time_transit=0.0,
            semimajor=semimajor_axis,
            impact_param=impact_param,
            radius=planet_rad,
            eccentricity=eccentricity,
            omega_peri=omega * ureg.rad,
            asc_node=Omega * ureg.rad,
            mass=0.0,
        )
        .bodies[0]
    )

    jaxoplanet_lc = 1 + limb_dark_light_curve(planet, u, order=100)(t)

    # generate comparison light curve
    Omega = jnp.arctan2(
        planet.sin_asc_node.to(ureg.radian).magnitude,
        planet.cos_asc_node.to(ureg.radian).magnitude,
    )
    Omega = jnp.where(Omega < 0, Omega + 2 * jnp.pi, Omega)

    omega = jnp.arctan2(
        planet.sin_omega_peri.to(ureg.radian).magnitude,
        planet.cos_omega_peri.to(ureg.radian).magnitude,
    )
    omega = jnp.where(omega < 0, omega + 2 * jnp.pi, omega)

    state = {
        "t_peri": planet.time_peri.to(ureg.day).magnitude,
        "times": TIMES.to(ureg.day).magnitude,
        "period": planet.period.to(ureg.day).magnitude,
        "a": planet.semimajor.to(ureg.R_sun).magnitude,
        "e": planet.eccentricity.to(ureg.dimensionless).magnitude,
        "i": planet.inclination.to(ureg.radian).magnitude,
        "Omega": Omega,
        "omega": omega,
        "f1": 0.0,  # always circular for testing
        "f2": 0.0,
        "r": planet.radius.to(ureg.R_sun).magnitude,
        "obliq": 0.0,
        "prec": 0.0,
        "ld_u_coeffs": jnp.array(u),
        "tidally_locked": False,
    }

    s = OblateSystem(**state)
    state = s._state

    test_lc = lightcurve(s._state, False)

    if not return_lc:
        m = (jaxoplanet_lc != 0) | (test_lc != 0)
        return (
            state,
            jnp.max(jnp.abs(jaxoplanet_lc - test_lc)),
            jnp.std(jnp.abs(jaxoplanet_lc[m] - test_lc[m])),
        )
    else:
        return state, jaxoplanet_lc, test_lc


def spherical_transit_compare(poly_limbdark_order):
    max_errs = []
    for i in tqdm(jnp.arange(N_JAXOPLANT_COMPARISONS)):
        key = jax.random.key(i)
        _, max_err, _ = light_curve_compare(key, poly_limbdark_order)
        max_errs.append(max_err)
    max_errs = jnp.array(max_errs)
    assert jnp.all(max_errs < 1e-7)


def test_spherical_transit():
    for p in POLY_ORDERS:
        spherical_transit_compare(p)
