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
from squishyplanet.engine.kepler import kepler, skypos

epsabs = epsrel = 1e-12



def _t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -1 + rho_00 - rho_x0 + rho_xx

def _t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -2*rho_xy + 2*rho_y0

def _t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -2 + 2*rho_00 - 2*rho_xx + 4*rho_yy

def _t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return 2*rho_xy + 2*rho_y0

def _t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -1 + rho_00 + rho_x0 + rho_xx

@jax.jit
def single_intersection_points(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00, **kwargs):
    t4 = _t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t3 = _t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t2 = _t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t1 = _t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t0 = _t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)

    polys = jnp.array([t4, t3, t2, t1, t0])
    roots = jnp.roots(polys, strip_zeros=False) # strip_zeros must be False to jit

    ts = jnp.where(jnp.imag(roots) == 0, jnp.real(roots), 999)
    xs = jnp.where(ts != 999, (1-ts**2)/(1+ts**2), ts)
    ys = jnp.where(ts != 999, 2*ts/(1+ts**2), ts)
    return xs, ys

# @jax.jit
# def multiple_intersection_points(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00, **kwargs):
#     t4 = _t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
#     t3 = _t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
#     t2 = _t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
#     t1 = _t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
#     t0 = _t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)

#     polys = jnp.array([t4, t3, t2, t1, t0]).T

#     roots = jax.vmap(lambda x: jnp.roots(x, strip_zeros=False))(polys)

#     ts = jnp.where(jnp.imag(roots) == 0, jnp.real(roots), 999)
#     xs = jnp.where(ts != 999, (1-ts**2)/(1+ts**2), ts)
#     ys = jnp.where(ts != 999, 2*ts/(1+ts**2), ts)
#     return xs, ys


@jax.jit
def planet_solution_vec(a, b, g_coeffs, c_x1, c_x2, c_x3, c_y1, c_y2, c_y3):

    def s0_integrand(s):
        return (jnp.cos(s) * c_x1 + jnp.sin(s) * c_x2 + c_x3) * (
            -(jnp.sin(s) * c_y1) + jnp.cos(s) * c_y2
        )

    def s1_integrand(s):
        return (
            (-(jnp.sin(s) * c_y1) + jnp.cos(s) * c_y2)
            * (
                jnp.pi
                + 6
                * (jnp.cos(s) * c_x1 + jnp.sin(s) * c_x2 + c_x3)
                * jnp.sqrt(
                    1
                    - (jnp.cos(s) * c_x1 + jnp.sin(s) * c_x2 + c_x3) ** 2
                    - (jnp.cos(s) * c_y1 + jnp.sin(s) * c_y2 + c_y3) ** 2
                )
                - 6
                * jnp.arctan(
                    (jnp.cos(s) * c_x1 + jnp.sin(s) * c_x2 + c_x3)
                    / jnp.sqrt(
                        1
                        - (jnp.cos(s) * c_x1 + jnp.sin(s) * c_x2 + c_x3) ** 2
                        - (jnp.cos(s) * c_y1 + jnp.sin(s) * c_y2 + c_y3) ** 2
                    )
                )
                * (-1 + (jnp.cos(s) * c_y1 + jnp.sin(s) * c_y2 + c_y3) ** 2)
            )
        ) / 12.0

    def sn_integrand(s):
        def scan_func(carry, scan_over):
            n = scan_over
            integrand = -(
                (
                    1
                    - (jnp.cos(s) * c_x1 + jnp.sin(s) * c_x2 + c_x3) ** 2
                    - (jnp.cos(s) * c_y1 + jnp.sin(s) * c_y2 + c_y3) ** 2
                )
                ** (n / 2.0)
                * (
                    c_x3 * (jnp.sin(s) * c_y1 - jnp.cos(s) * c_y2)
                    + c_x2 * (c_y1 + jnp.cos(s) * c_y3)
                    - c_x1 * (c_y2 + jnp.sin(s) * c_y3)
                )
            )
            return None, integrand

        integrand = jax.lax.scan(
            scan_func, None, jnp.arange(g_coeffs.shape[0])[2:]
        )[1]
        return integrand

    higher_terms, _ = quadgk(
        sn_integrand, jnp.array([a, b]), epsabs=epsabs, epsrel=epsrel
    )

    s0, _ = quadgk(
        s0_integrand, jnp.array([a, b]), epsabs=epsabs, epsrel=epsrel
    )
    s1, _ = quadgk(
        s1_integrand, jnp.array([a, b]), epsabs=epsabs, epsrel=epsrel
    )

    s0 = jnp.array(
        [s0]
    )  # needed b/c when scanning over individual phases, this will return a scalar
    s1 = jnp.array([s1])
    return jnp.concatenate((s0, s1, higher_terms))


@jax.jit
def star_solution_vec(a, b, g_coeffs, c_x1, c_x2, c_x3, c_y1, c_y2, c_y3):
    x1 = c_x1 * jnp.cos(a) + c_x2 * jnp.sin(a) + c_x3
    y1 = c_y1 * jnp.cos(a) + c_y2 * jnp.sin(a) + c_y3
    _theta1 = jnp.arctan2(y1, x1)
    _theta1 = jnp.where(_theta1 < 0, _theta1 + 2 * jnp.pi, _theta1)
    #jax.debug.print("theta1: {x}", x=_theta1)

    x2 = c_x1 * jnp.cos(b) + c_x2 * jnp.sin(b) + c_x3
    y2 = c_y1 * jnp.cos(b) + c_y2 * jnp.sin(b) + c_y3
    _theta2 = jnp.arctan2(y2, x2)
    _theta2 = jnp.where(_theta2 < 0, _theta2 + 2 * jnp.pi, _theta2)
    #jax.debug.print("theta2: {x}", x=_theta2)

    theta1 = jnp.where(_theta1 < _theta2, _theta1, _theta2)
    theta2 = jnp.where(_theta1 < _theta2, _theta2, _theta1)

    delta = jnp.abs(
        jnp.arctan2(jnp.sin(theta1 - theta2), jnp.cos(theta1 - theta2))
    )
    #jax.debug.print("delta 1: {x}", x=delta)
    delta = theta2 - theta1
    #jax.debug.print("delta 2: {x}", x=delta)

    s0_integrand = lambda t: jnp.cos(t) ** 2
    s1_integrand = lambda t: jnp.where(
        (t < jnp.pi / 2) | (t > 3 * jnp.pi / 2),
        (jnp.pi * jnp.cos(t) * (5 + 3 * jnp.cos(2 * t))) / 24.0,
        -(jnp.pi * jnp.cos(t) * (1 + 3 * jnp.cos(2 * t))) / 24.0,
    )

    def no_wrap(delta):
        s0, _ = quadgk(
            s0_integrand,
            jnp.array([theta1, theta2]),
            epsabs=epsabs,
            epsrel=epsrel,
        )
        s1, _ = quadgk(
            s1_integrand,
            jnp.array([theta1, theta2]),
            epsabs=epsabs,
            epsrel=epsrel,
        )
        return s0, s1

    def wrap(delta):
        s0, _ = quadgk(
            s0_integrand,
            jnp.array([theta2, 2 * jnp.pi]),
            epsabs=epsabs,
            epsrel=epsrel,
        )
        s0 += quadgk(
            s0_integrand, jnp.array([0, theta1]), epsabs=epsabs, epsrel=epsrel
        )[0]
        s1, _ = quadgk(
            s1_integrand,
            jnp.array([theta2, 2 * jnp.pi]),
            epsabs=epsabs,
            epsrel=epsrel,
        )
        s1 += quadgk(
            s1_integrand, jnp.array([0, theta1]), epsabs=epsabs, epsrel=epsrel
        )[0]
        return s0, s1

    s0, s1 = jax.lax.cond(delta < jnp.pi, no_wrap, wrap, delta)
    #jax.debug.print("s0: {x}", x=s0)
    #jax.debug.print("s1: {x}", x=s1)

    solution_vec = jnp.zeros(g_coeffs.shape[0])
    solution_vec = solution_vec.at[0].set(s0)
    solution_vec = solution_vec.at[1].set(s1)
    return solution_vec


@jax.jit
def lightcurve(state):

    # array we'll modify if the planet is in transit
    fluxes = jnp.ones_like(state["times"])

    time_deltas = state["times"] - state["t_peri"]
    mean_anomalies = 2 * jnp.pi * time_deltas / state["period"]
    true_anomalies = kepler(mean_anomalies, state["e"])
    state["f"] = true_anomalies

    state["prec"] = jnp.where(state["tidally_locked"], state["f"], state["prec"])

    # the coefficients of the implicit 3d surface
    three = planet_3d_coeffs(**state)
    # the coefficients of the implicit 2d surface
    two = planet_2d_coeffs(**three)
    # the coefficients of the parametric projected ellipse
    para = poly_to_parametric(**two)

    # convert the u coefficients to g coefficients
    u_coeffs = jnp.ones(state["ld_u_coeffs"].shape[0] + 1) * (-1)
    u_coeffs = u_coeffs.at[1:].set(state["ld_u_coeffs"])
    g_coeffs = jnp.matmul(state["greens_basis_transform"], u_coeffs)

    # total flux from the star. 1/eq. 28 in Agol, Luger, and Foreman-Mackey 2020
    # note, multiply, don't divide
    normalization_constant = 1 / (
        jnp.pi * (g_coeffs[0] + (2 / 3) * g_coeffs[1])
    )

    # cartesian position of the planet at each timestep
    positions = skypos(**state)

    # boolean mask, is the planet transiting at each timestep?
    # possibly_in_transit = (
    #     (jnp.abs(positions[0, :]) < 1.0 + state["r"])
    #     * (jnp.abs(positions[1, :]) < 1.0 + state["r"])
    #     * (positions[2, :] > 0)
    # )
    possibly_in_transit = (
        positions[0, :] ** 2 + positions[1, :] ** 2
        <= (1.0 + state["r"] * 1.1) ** 2
    ) * (positions[2, :] > 0)

    def not_on_limb(X):
        para, _, _ = X

        def fully_transiting(para):
            solution_vectors = planet_solution_vec(
                a=0.0, b=2 * jnp.pi, g_coeffs=g_coeffs, **para
            )
            blocked_flux = (
                jnp.matmul(g_coeffs, solution_vectors) * normalization_constant
            )
            #jax.debug.print("s0: {x}", x=solution_vectors[0])
            #jax.debug.print("s1: {x}", x=solution_vectors[1])
            #jax.debug.print("s2: {x}", x=solution_vectors[1])
            #jax.debug.print("g0: {x}", x=g_coeffs[0])
            #jax.debug.print("g1: {x}", x=g_coeffs[1])
            #jax.debug.print("g2: {x}", x=g_coeffs[2])

            #jax.debug.print("fully transiting")
            #jax.debug.print(
                # "normalization constant: {x}", x=normalization_constant
            # )
            #jax.debug.print("g coeffs: {x}", x=g_coeffs)
            #jax.debug.print("planet solution vector: {x}", x=solution_vectors)
            #jax.debug.print("planet contribution: {x}", x=blocked_flux)
            #jax.debug.print("total blocked: {x}\n", x=blocked_flux)

            return blocked_flux

        def not_transiting(para):
            return 0.0

        #jax.debug.print("center2: {x}", x=para["c_x3"] ** 2 + para["c_y3"] ** 2)
        return jax.lax.cond(
            para["c_x3"] ** 2 + para["c_y3"] ** 2 <= 1,
            fully_transiting,
            not_transiting,
            para,
        )

    def partially_transiting(X):
        para, xs, ys = X
        #jax.debug.print("xs: {x}", x=xs)
        #jax.debug.print("ys: {x}", x=ys)
        # q = jnp.linalg.norm(jnp.array([xs, ys]))
        #jax.debug.print("transiting? {x}", x=q < (1 + state["r"]))
        alphas = cartesian_intersection_to_parametric_angle(xs, ys, **para)
        alphas = jnp.where(xs != 999, alphas, 2 * jnp.pi)
        alphas = jnp.where(alphas < 0, alphas + 2 * jnp.pi, alphas)
        alphas = jnp.where(alphas > 2 * jnp.pi, alphas - 2 * jnp.pi, alphas)
        alphas = jnp.sort(alphas)

        test_ang = alphas[0] + (alphas[1] - alphas[0]) / 2
        test_ang = jnp.where(
            test_ang > 2 * jnp.pi, test_ang - 2 * jnp.pi, test_ang
        )

        _x = (
            para["c_x1"] * jnp.cos(test_ang)
            + para["c_x2"] * jnp.sin(test_ang)
            + para["c_x3"]
        )
        _y = (
            para["c_y1"] * jnp.cos(test_ang)
            + para["c_y2"] * jnp.sin(test_ang)
            + para["c_y3"]
        )
        test_val = jnp.sqrt(_x**2 + _y**2)

        def testval_inside_star(_):
            #jax.debug.print("inside")
            solution_vectors = planet_solution_vec(
                alphas[0], alphas[1], g_coeffs, **para
            )
            #jax.debug.print("planet solution: {x}", x=solution_vectors)
            planet_contribution = (
                jnp.matmul(solution_vectors, g_coeffs) * normalization_constant
            )
            return planet_contribution

        def testval_outside_star(_):
            #jax.debug.print("outside")
            # print(alphas)
            leg1_solution_vec = planet_solution_vec(
                alphas[1], 2 * jnp.pi, g_coeffs, **para
            )
            #jax.debug.print("leg 1 solution: {x}", x=leg1_solution_vec)
            leg1 = jnp.matmul(leg1_solution_vec, g_coeffs)
            leg2_solution_vec = planet_solution_vec(
                0.0, alphas[0], g_coeffs, **para
            )
            #jax.debug.print("leg 2 solution: {x}", x=leg2_solution_vec)
            leg2 = jnp.matmul(leg2_solution_vec, g_coeffs)
            planet_contribution = (leg1 + leg2) * normalization_constant
            return planet_contribution

        planet_contribution = jax.lax.cond(
            test_val > 1, testval_outside_star, testval_inside_star, ()
        )

        star_solution_vectors = star_solution_vec(
            alphas[0], alphas[1], g_coeffs, **para
        )
        star_contribution = (
            jnp.matmul(star_solution_vectors, g_coeffs)
            * normalization_constant
        )

        #jax.debug.print("partially transiting")
        #jax.debug.print(
            # "normalization constant: {x}", x=normalization_constant
        # )
        #jax.debug.print("g coeffs: {x}", x=g_coeffs)
        #jax.debug.print("alphas: {x}", x=alphas)
        #jax.debug.print("star solution: {x}", x=star_solution_vectors)
        #jax.debug.print("star contribution: {x}", x=star_contribution)
        #jax.debug.print("planet contribution: {x}", x=planet_contribution)

        total_blocked = planet_contribution + star_contribution
        #jax.debug.print("total blocked: {x}\n", x=total_blocked)

        return total_blocked

    def transiting(X):
        indv_para, indv_two = X
        (
            xs,
            ys,
        ) = single_intersection_points(**indv_two)
        #jax.debug.print("transiting decision func")
        #jax.debug.print("para: {x}", x=indv_para)
        #jax.debug.print("two: {x}", x=indv_two)
        # #jax.debug.print("")
        #jax.debug.print("xs: {x}", x=xs)
        #jax.debug.print("ys: {x}", x=ys)
        on_limb = jnp.sum(xs) != 999 * 4
        #jax.debug.print("on_limb: {x}", x=on_limb)

        return jax.lax.cond(
            on_limb,
            partially_transiting,
            not_on_limb,
            (indv_para, xs, ys),
        )

    def not_transiting(X):
        return 0.0

    def scan_func(carry, scan_over):
        indv_para, indv_two, mask = scan_over
        return None, jax.lax.cond(
            mask, transiting, not_transiting, (indv_para, indv_two)
        )

    # if prec isn't the same length as f, we've actually made
    # it to this point with some of the three, two vectors being
    # the same length as f, and the others are just scalars
    # if prec isn't changing, the planet's orientation isn't either,
    # so none of your quadratic terms vary
    if state["prec"].shape != true_anomalies.shape:
        two["rho_xx"] = jnp.ones_like(state["f"]) * two["rho_xx"]
        two["rho_xy"] = jnp.ones_like(state["f"]) * two["rho_xy"]
        two["rho_yy"] = jnp.ones_like(state["f"]) * two["rho_yy"]

    transit_fluxes = jax.lax.scan(
        scan_func, None, (para, two, possibly_in_transit), None
    )[1]

    return fluxes - transit_fluxes
