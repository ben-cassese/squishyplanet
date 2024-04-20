# this is a quadratic-only limb darkening model, uses a different method than the
# Agol, Luger, Foreman-Mackey 2020 scheme. Should produce the same results as the
# more generate polynomial_limb_darkened_transit.py model when all terms > u2 are 0

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from quadax import quadgk

from squishyplanet.engine.planet_3d import planet_3d_coeffs
from squishyplanet.engine.planet_2d import planet_2d_coeffs
from squishyplanet.engine.parametric_ellipse import (
    poly_to_parametric,
    cartesian_intersection_to_parametric_angle,
)
from squishyplanet.engine.skypos import skypos
from squishyplanet.engine.star_planet_intersection import single_intersection_points
from squishyplanet.engine.kepler import kepler


@jax.jit
def planet_flux_integrand(alpha, u1, u2, c_x1, c_x2, c_x3, c_y1, c_y2, c_y3, **kwargs):
    return (
        (
            c_x3 * (-(jnp.sin(alpha) * c_y1) + jnp.cos(alpha) * c_y2)
            - c_x2 * (c_y1 + jnp.cos(alpha) * c_y3)
            + c_x1 * (c_y2 + jnp.sin(alpha) * c_y3)
        )
        * (
            3
            * u2
            * (
                (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3) ** 2
                + (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3) ** 2
            )
            ** 2
            - 4
            * (u1 + 2 * u2)
            * (
                -1
                + jnp.sqrt(
                    1
                    - (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3) ** 2
                    - (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3) ** 2
                )
            )
            + 2
            * (
                (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3) ** 2
                + (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3) ** 2
            )
            * (
                3
                - 3 * u1
                - 6 * u2
                + 2
                * u1
                * jnp.sqrt(
                    1
                    - (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3) ** 2
                    - (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3) ** 2
                )
                + 4
                * u2
                * jnp.sqrt(
                    1
                    - (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3) ** 2
                    - (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3) ** 2
                )
            )
        )
    ) / (
        12.0
        * (
            (jnp.cos(alpha) * c_x1 + jnp.sin(alpha) * c_x2 + c_x3) ** 2
            + (jnp.cos(alpha) * c_y1 + jnp.sin(alpha) * c_y2 + c_y3) ** 2
        )
    )


# @jax.jit
def star_flux_integral(alpha1, alpha2, u1, u2, c_x1, c_x2, c_x3, c_y1, c_y2, c_y3):
    x1 = c_x1 * jnp.cos(alpha1) + c_x2 * jnp.sin(alpha1) + c_x3
    y1 = c_y1 * jnp.cos(alpha1) + c_y2 * jnp.sin(alpha1) + c_y3
    theta1 = jnp.arctan2(y1, x1)
    theta1 = jnp.where(theta1 < 0, theta1 + 2 * jnp.pi, theta1)

    x2 = c_x1 * jnp.cos(alpha2) + c_x2 * jnp.sin(alpha2) + c_x3
    y2 = c_y1 * jnp.cos(alpha2) + c_y2 * jnp.sin(alpha2) + c_y3
    theta2 = jnp.arctan2(y2, x2)
    theta2 = jnp.where(theta2 < 0, theta2 + 2 * jnp.pi, theta2)

    # print(theta1, theta2)
    # jax.debug.print("theta1 {x}", x=theta1)
    # jax.debug.print("theta2 {x}", x=theta2)

    delta = jnp.abs(jnp.arctan2(jnp.sin(theta1 - theta2), jnp.cos(theta1 - theta2)))
    # jax.debug.print("weird delta {x}", x=delta)

    # jax.debug.print("star contribution {x}", x=delta / (2 * jnp.pi))
    return delta / (2 * jnp.pi)


def limb_darkening_profile(x, y, u1, u2, **kwargs):
    return (
        1
        + u1 * (-1 + jnp.sqrt(1 - x**2 - y**2))
        - u2 * (-1 + jnp.sqrt(1 - x**2 - y**2)) ** 2
    )


@jax.jit
def lightcurve(state, times):
    epsabs = epsrel = 1e-12

    # kepler's eq for times -> f
    time_deltas = times - state["t_peri"]
    mean_anomalies = 2 * jnp.pi * time_deltas / state["period"]
    f = kepler(mean_anomalies, state["e"])
    state["f"] = f
    # state["f"] = times

    three = planet_3d_coeffs(**state)
    two = planet_2d_coeffs(**three)
    para = poly_to_parametric(**two)

    fluxes = jnp.ones_like(times)

    positions = skypos(**state)

    possibly_in_transit = (
        (jnp.abs(positions[0, :]) < 1.0 + state["r"])
        * (jnp.abs(positions[1, :]) < 1.0 + state["r"])
        * (positions[2, :] > 0)
    )

    # not thrilled with this implementation- you can feed all of para straight to
    # quadgk without scanning or vmapping, which makes me think you should be able to
    # feed it just all of the entries where it's transiting. but, I kept running into
    # ConcretizationTypeError issues, so this is the workaround for now
    def fully_transiting(X):
        para, _, _ = X
        func = jax.tree_util.Partial(
            planet_flux_integrand, u1=state["u1"], u2=state["u2"], **para
        )
        blocked_flux = quadgk(func, [0, 2 * jnp.pi], epsabs=epsabs, epsrel=epsrel)[0]
        blocked_flux = blocked_flux / (
            -(1 / 6) * jnp.pi * (-6 + 2 * state["u1"] + state["u2"])
        )
        return blocked_flux

    def partially_transiting(X):
        # return jnp.nan
        para, xs, ys = X
        alphas = cartesian_intersection_to_parametric_angle(xs, ys, **para)
        alphas = jnp.where(xs != 999, alphas, 2 * jnp.pi)
        alphas = jnp.where(alphas < 0, alphas + 2 * jnp.pi, alphas)
        alphas = jnp.where(alphas > 2 * jnp.pi, alphas - 2 * jnp.pi, alphas)
        alphas = jnp.sort(alphas)
        # jax.debug.print("xs {x}", x=xs)
        # jax.debug.print("ys {x}", x=ys)
        # jax.debug.print("alphas {x}", x=alphas)

        test_ang = alphas[0] + (alphas[1] - alphas[0]) / 2
        test_ang = jnp.where(test_ang > 2 * jnp.pi, test_ang - 2 * jnp.pi, test_ang)
        # jax.debug.print("test_ang {x}", x=test_ang)

        test_val = limb_darkening_profile(
            x=para["c_x1"] * jnp.cos(test_ang)
            + para["c_x2"] * jnp.sin(test_ang)
            + para["c_x3"],
            y=para["c_y1"] * jnp.cos(test_ang)
            + para["c_y2"] * jnp.sin(test_ang)
            + para["c_y3"],
            u1=0.6,
            u2=0.2,
            **para,
        )
        # jax.debug.print("test_val {x}", x=test_val)

        func = jax.tree_util.Partial(
            planet_flux_integrand, u1=state["u1"], u2=state["u2"], **para
        )
        # full = quadgk(func, [0, 2 * jnp.pi], epsabs=epsabs, epsrel=epsrel)[
        #     0
        # ] / (-(1 / 6) * jnp.pi * (-6 + 2 * state["u1"] + state["u2"]))

        def testval_is_not_nan(_):
            planet_contribution = quadgk(
                func, [alphas[0], alphas[1]], epsabs=epsabs, epsrel=epsrel
            )[0] / (-(1 / 6) * jnp.pi * (-6 + 2 * state["u1"] + state["u2"]))
            return planet_contribution

        def testval_is_nan(_):
            # planet_contribution =

            leg1 = quadgk(func, [alphas[1], 2 * jnp.pi], epsabs=epsabs, epsrel=epsrel)[
                0
            ]
            leg2 = quadgk(func, [0.0, alphas[0]], epsabs=epsabs, epsrel=epsrel)[0]
            planet_contribution = (leg1 + leg2) / (
                -(1 / 6) * jnp.pi * (-6 + 2 * state["u1"] + state["u2"])
            )
            return planet_contribution

        planet_contribution = jax.lax.cond(
            jnp.isnan(test_val), testval_is_nan, testval_is_not_nan, ()
        )
        # jax.debug.print("planet_contribution {x}", x=planet_contribution)
        # #jax.debug.print("full {x}", x=full)
        # #jax.debug.print("side1 {x}", x=side1)
        # #jax.debug.print("side2 {x}", x=side2)

        # planet_contribution = jnp.where(~jnp.isnan(test_val), side2, side1)
        star_contribution = star_flux_integral(
            alpha1=alphas[0],
            alpha2=alphas[1],
            u1=state["u1"],
            u2=state["u2"],
            **para,
        )

        total_blocked = planet_contribution + star_contribution

        return total_blocked

    def transiting(X):
        indv_para, indv_two = X
        (
            xs,
            ys,
        ) = single_intersection_points(**indv_two)
        on_limb = jnp.sum(xs) != 999 * 4

        return jax.lax.cond(
            on_limb,
            partially_transiting,
            fully_transiting,
            (indv_para, xs, ys),
        )

    def not_transiting(X):
        return 0.0

    def scan_func(carry, scan_over):
        indv_para, indv_two, mask = scan_over
        return None, jax.lax.cond(
            mask, transiting, not_transiting, (indv_para, indv_two)
        )

    # _, transit_fluxes = jax.lax.scan(
    #     scan_func, None, (para, two, possibly_in_transit), None
    # )
    # fluxes -= transit_fluxes
    two["rho_xx"] = jnp.ones_like(f) * two["rho_xx"]
    two["rho_xy"] = jnp.ones_like(f) * two["rho_xy"]
    two["rho_yy"] = jnp.ones_like(f) * two["rho_yy"]

    transit_fluxes = jax.lax.scan(
        scan_func, None, (para, two, possibly_in_transit), None
    )[1]

    return fluxes - transit_fluxes
