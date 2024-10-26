import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from quadax import quadgk
from functools import partial

from squishyplanet.engine.planet_3d import planet_3d_coeffs
from squishyplanet.engine.planet_2d import planet_2d_coeffs
from squishyplanet.engine.parametric_ellipse import (
    poly_to_parametric,
    cartesian_intersection_to_parametric_angle,
)
from squishyplanet.engine.kepler import kepler, skypos, t0_to_t_peri

epsabs = epsrel = 1e-12


def _t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -1 + rho_00 - rho_x0 + rho_xx


def _t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -2 * rho_xy + 2 * rho_y0


def _t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -2 + 2 * rho_00 - 2 * rho_xx + 4 * rho_yy


def _t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return 2 * rho_xy + 2 * rho_y0


def _t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00):
    return -1 + rho_00 + rho_x0 + rho_xx


@jax.jit
def _single_intersection_points(
    rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00, **kwargs
):
    t4 = _t4(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t3 = _t3(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t2 = _t2(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t1 = _t1(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)
    t0 = _t0(rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00)

    polys = jnp.array([t4, t3, t2, t1, t0])
    roots = jnp.roots(polys, strip_zeros=False)  # strip_zeros must be False to jit

    ts = jnp.where(jnp.imag(roots) == 0, jnp.real(roots), 999)
    xs = jnp.where(ts != 999, (1 - ts**2) / (1 + ts**2), ts)
    ys = jnp.where(ts != 999, 2 * ts / (1 + ts**2), ts)
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
def parameterize_2d_helper(projected_r, projected_f, projected_theta, xc, yc):
    """
    Convert from the alternative sky-projected parameterization to the same format
    used by the 3D parameterization.

    A good chunk of the code assumes that the planet's center is determined by the
    orbital elements and that it's outline is derived from an equatorial radius ``r``,
    a z-flattening ``f1``, a y-flattening ``f2``, and two body-centered rotations
    ``obliq`` and ``prec``. This are useful to have when working with phase curves that
    are sensitive to the actual 3D orientation of the planet, but when dealing with
    transits only, this parameterization is overkill and allows a bunch of degeneracies.
    So, if only doing transits, it is more convenient to parameterize the planet by
    its projected radius in the x and y directions, and the angle of the projected
    ellipse. This function takes in those parameters and returns the same dictionaries
    you'd get if you fed a full 3D parameterization into
    :func:`planet_2d.planet_2d_coeffs`.

    Args:
        projected_r (float): The projected "x" radius of the planet.
        projected_f (float): The flattening of the projected ellipse.
        projected_theta (float): The angle of the projected ellipse.

    Returns:
        tuple:
            A tuple of two dictionaries. The first dictionary contains the coefficients
            of the quadratic equation that describes the projected ellipse. The second
            dictionary contains coefficients that describe the parametric form of that
            same ellipse.

    """
    # projected_
    projected_r2 = projected_r * (1 - projected_f)
    cos_t = jnp.cos(projected_theta)
    sin_t = jnp.sin(projected_theta)

    two = {
        "rho_xx": cos_t**2 / projected_r**2 + sin_t**2 / projected_r2**2,
        "rho_xy": (2 * cos_t * sin_t) / projected_r**2
        - (2 * cos_t * sin_t) / projected_r2**2,
        "rho_x0": (-2 * cos_t**2 * xc) / projected_r**2
        - (2 * cos_t * yc * sin_t) / projected_r**2
        + (2 * cos_t * yc * sin_t) / projected_r2**2
        - (2 * xc * sin_t**2) / projected_r2**2,
        "rho_yy": cos_t**2 / projected_r2**2 + sin_t**2 / projected_r**2,
        "rho_y0": (-2 * cos_t**2 * yc) / projected_r2**2
        - (2 * cos_t * xc * sin_t) / projected_r**2
        + (2 * cos_t * xc * sin_t) / projected_r2**2
        - (2 * yc * sin_t**2) / projected_r**2,
        "rho_00": (cos_t**2 * xc**2) / projected_r**2
        + (cos_t**2 * yc**2) / projected_r2**2
        + (2 * cos_t * xc * yc * sin_t) / projected_r**2
        - (2 * cos_t * xc * yc * sin_t) / projected_r2**2
        + (yc**2 * sin_t**2) / projected_r**2
        + (xc**2 * sin_t**2) / projected_r2**2,
    }

    # para = {
    #     "c_x1": jnp.ones_like(xc) * (projected_r * cos_t),
    #     "c_x2": jnp.ones_like(xc) * (-projected_r2 * sin_t),
    #     "c_x3": xc,
    #     "c_y1": jnp.ones_like(xc) * (projected_r * sin_t),
    #     "c_y2": jnp.ones_like(xc) * (projected_r2 * cos_t),
    #     "c_y3": yc,
    # }
    para = poly_to_parametric(**two)

    return two, para


@jax.jit
def planet_solution_vec(a, b, g_coeffs, c_x1, c_x2, c_x3, c_y1, c_y2, c_y3):
    """
    Compute the "solution vector" for a 1D path across the star that lies on the outline
    of the planet.

    This computes Eq. 21 of `Agol, Luger, and Foreman-Mackey 2020
    <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_. But, instead of
    doing it analytically, this uses the ``quadax`` package to numerically solve the
    required integrals. For terms s_2 and higher, this is straightforward to do based on
    the equations in the paper: we simply parameterize the outline of the planet by
    some angle :math:`\\alpha`, then numerically integrate the dot product of Eq. 62
    with that parameterization between the two endpoints of the path. For the first two
    lower-order terms however, Agol et al. do not provide an equivalent of Eq. 62 and
    instead provide only the analytic solutions. We therefore use the following as the
    equivalents for Eq. 62 for these terms:

    .. math::

        G_0 = \\{0, x\\}

    .. math::

        G_1 = \\left\\{0, \\frac{1}{2} \\left(x \\sqrt{-x^2-y^2+1}-\\left(y^2-1\\right) \\tan ^{-1}\\left(\\frac{x}{\\sqrt{-x^2-y^2+1}}\\right)\\right)+\\frac{\\pi }{12} \\right\\}

    These expressions were derived by solving the required PDE in Eq. 14 with the
    boundary conditions from Eq. 27. Finally, the C coefficients here describe the
    parametric form of the planet's outline as seen by the observer, and they satisfy:

    .. math::

        x = c_{x1} \\cos(\\alpha) + c_{x2} \\sin(\\alpha) + c_{x3}

        y = c_{y1} \\cos(\\alpha) + c_{y2} \\sin(\\alpha) + c_{y3}

    for some angle :math:`\\alpha \\in [0, 2\\pi)`.

    Args:
        a (float):
            The starting parameter for the path along the planet's outline,
            :math:`\\alpha_0`.
        b (float):
            The ending parameter for the path along the planet's outline,
            :math:`\\alpha_1`.
        g_coeffs (Array):
            The system-specific limb darkening coefficients in the Green's basis.
            Computed by multiplying the u coefficients with the change of basis matrix
            from :func:`greens_basis_transform.generate_change_of_basis_matrix`.
        c_x1 (float):
            The first coefficient of the parametric 2D outline of the planet.
        c_x2 (float):
            The second coefficient of the parametric 2D outline of the planet.
        c_x3 (float):
            The third coefficient of the parametric 2D outline of the planet.
        c_y1 (float):
            The fourth coefficient of the parametric 2D outline of the planet.
        c_y2 (float):
            The fifth coefficient of the parametric 2D outline of the planet.
        c_y3 (float):
            The sixth coefficient of the parametric 2D outline of the planet.

    Returns:
        Array:
            The solution vector for the path along the planet's outline. The shape will
            match that of the input ``g_coeffs``.
    """

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

        integrand = jax.lax.scan(scan_func, None, jnp.arange(g_coeffs.shape[0])[2:])[1]
        return integrand

    higher_terms, _ = quadgk(
        sn_integrand, jnp.array([a, b]), epsabs=epsabs, epsrel=epsrel
    )

    s0, _ = quadgk(s0_integrand, jnp.array([a, b]), epsabs=epsabs, epsrel=epsrel)
    s1, _ = quadgk(s1_integrand, jnp.array([a, b]), epsabs=epsabs, epsrel=epsrel)

    s0 = jnp.array(
        [s0]
    )  # needed b/c when scanning over individual phases, this will return a scalar
    s1 = jnp.array([s1])
    return jnp.concatenate((s0, s1, higher_terms))


@jax.jit
def star_solution_vec(a, b, g_coeffs, c_x1, c_x2, c_x3, c_y1, c_y2, c_y3):
    """
    Compute the "solution vector" for a 1D path across the star that lies on the edge
    of the star itself.

    This is equivalent to :func:`planet_solution_vec`, but instead of integrating over
    paths that lie on the planet's outline, we integrate over paths that lie on the edge
    of the star. As pointed out in the paragraph following Eq. 69 in `Agol, Luger, and
    Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_,
    the contribution of all terms higher than :math:`G_1` will be zero in this case
    since we have limited ourselves to :math:`z=0` by remaining on the star's boundary.
    This simplifies things somewhat, though we do still have to numerically integrate
    the dot product of the parametric form of the star's outline with the :math:`G_0`
    and :math:`G_1` terms written out in :func:`planet_solution_vec`. Technically we
    probably could use the analytic solutions for these terms, but so far we have not.

    Args:
        a (float):
            The starting parameter for the path along the star's outline,
            :math:`\\alpha_0`. Note: here :math:`\\alpha` is the angle parameterizing
            the path on the *planet's* outline, not the star's, even though the path
            we will integrate over is on the star. We convert to the relevant parameters
            internally.
        b (float):
            The ending parameter for the path along the star's outline,
            :math:`\\alpha_1`.
        g_coeffs (Array):
            The system-specific limb darkening coefficients in the Green's basis.
            Computed by multiplying the u coefficients with the change of basis matrix
            from :func:`greens_basis_transform.generate_change_of_basis_matrix`.
        c_x1 (float):
            The first coefficient of the parametric 2D outline of the planet.
        c_x2 (float):
            The second coefficient of the parametric 2D outline of the planet.
        c_x3 (float):
            The third coefficient of the parametric 2D outline of the planet.
        c_y1 (float):
            The fourth coefficient of the parametric 2D outline of the planet.
        c_y2 (float):
            The fifth coefficient of the parametric 2D outline of the planet.
        c_y3 (float):
            The sixth coefficient of the parametric 2D outline of the planet.

    Returns:
        Array:
            The solution vector for the path along the planet's outline. The shape will
            match that of the input ``g_coeffs``.

    """
    x1 = c_x1 * jnp.cos(a) + c_x2 * jnp.sin(a) + c_x3
    y1 = c_y1 * jnp.cos(a) + c_y2 * jnp.sin(a) + c_y3
    _theta1 = jnp.arctan2(y1, x1)
    _theta1 = jnp.where(_theta1 < 0, _theta1 + 2 * jnp.pi, _theta1)

    x2 = c_x1 * jnp.cos(b) + c_x2 * jnp.sin(b) + c_x3
    y2 = c_y1 * jnp.cos(b) + c_y2 * jnp.sin(b) + c_y3
    _theta2 = jnp.arctan2(y2, x2)
    _theta2 = jnp.where(_theta2 < 0, _theta2 + 2 * jnp.pi, _theta2)

    theta1 = jnp.where(_theta1 < _theta2, _theta1, _theta2)
    theta2 = jnp.where(_theta1 < _theta2, _theta2, _theta1)

    delta = jnp.abs(jnp.arctan2(jnp.sin(theta1 - theta2), jnp.cos(theta1 - theta2)))
    delta = theta2 - theta1

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

    solution_vec = jnp.zeros(g_coeffs.shape[0])
    solution_vec = solution_vec.at[0].set(s0)
    solution_vec = solution_vec.at[1].set(s1)
    return solution_vec


@partial(jax.jit, static_argnames=("parameterize_with_projected_ellipse",))
def lightcurve(state, parameterize_with_projected_ellipse):
    """
    The main function for computing a transit light curve.

    This function will return a 1-D array representing the flux recieved from the star,
    where each entry corresponds to a time in the input `state` dictionary. It first
    transforms the `state` into the implicit 3D surface of the planet, the implicit 2D
    sky-projected outline of the planet, and a parametric form of that outline for each
    time step. These are vectorized operations that are computed simulataneously across
    all times. It then solves for the intersection points of the planet and star, and
    if the planet is either partially or fully transiting, numerically solves the
    required 1D integrals that leverage Green's Theorem to compute the blocked flux. The
    flux-blocking calculations are done sequentially for each timestep using
    ``jax.lax.scan``, which seemed to be more efficient than vectorizing again while
    switching between braches with something like ``jax.lax.cond``. Keep these different
    behaviors in mind when computing dense lightcurves with ~100s of thousands of time
    steps: the first part will require enough memory to compute and store ~30 values for
    each step, but then the actual 1D integrals will be computed sequentially.

    Args:
        state (dict):
            A dictionary containing all of the keys that are included in an
            :func:`OblateSystem` ``state`` attribute.
        parameterize_with_projected_ellipse (bool):
            If ``True``, the planet's outline will be parameterized by the projected
            ellipse as seen by the observer. If ``False``, the planet's outline will be
            set by the full 3D parameterization of the planet. When dealing with planets
            that are not tidally locked and/or far from their host star and/or very
            close to spherical, you won't be able to tell the difference between these
            two parameterizations since the projected area won't be changing. In that
            case, it's better to use the simpler 2D parameterization to avoid the
            degeneracies and extra computation that can arise from the 3D
            parameterization. This argument is static for the JIT-compiled function.

    Returns:
        Array:
            The flux received from the star at each time step for the times included as
            ``state["times"]``.

    """

    # array we'll modify if the planet is in transit
    fluxes = jnp.ones_like(state["times"])

    if state["t0"] is not None:
        state["t_peri"] = t0_to_t_peri(**state)

    time_deltas = state["times"] - state["t_peri"]
    mean_anomalies = 2 * jnp.pi * time_deltas / state["period"]
    true_anomalies = kepler(mean_anomalies, state["e"])
    state["f"] = true_anomalies

    # convert the u coefficients to g coefficients
    u_coeffs = jnp.ones(state["ld_u_coeffs"].shape[0] + 1) * (-1)
    u_coeffs = u_coeffs.at[1:].set(state["ld_u_coeffs"])
    g_coeffs = jnp.matmul(state["greens_basis_transform"], u_coeffs)

    # total flux from the star. 1/eq. 28 in Agol, Luger, and Foreman-Mackey 2020
    # note, multiply, don't divide
    normalization_constant = 1 / (jnp.pi * (g_coeffs[0] + (2 / 3) * g_coeffs[1]))

    # cartesian position of the planet at each timestep
    positions = skypos(**state)

    if parameterize_with_projected_ellipse:
        area = jnp.pi * state["projected_effective_r"] ** 2
        r1 = jnp.sqrt(area / ((1 - state["projected_f"]) * jnp.pi))
        r2 = r1 * (1 - state["projected_f"])
        two, para = parameterize_2d_helper(
            r1,
            state["projected_f"],
            state["projected_theta"],
            positions[0, :],
            positions[1, :],
        )
        # force the shapes to match: if user inputs a scaler for one value but the
        # others are still (1,) there'd be a problem. all is fine if they're all
        # scalars or all arrays though
        r1 = jnp.ones_like(r2) * r1
        largest_r = jnp.max(jnp.array([r1, r2]))

    else:
        state["prec"] = jnp.where(state["tidally_locked"], state["f"], state["prec"])

        # the coefficients of the implicit 3d surface
        three = planet_3d_coeffs(**state)
        # the coefficients of the implicit 2d surface
        two = planet_2d_coeffs(**three)
        # the coefficients of the parametric projected ellipse
        para = poly_to_parametric(**two)

        largest_r = state["r"]

    possibly_in_transit = (
        positions[0, :] ** 2 + positions[1, :] ** 2 <= (1.0 + largest_r * 1.1) ** 2
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

            return blocked_flux

        def not_transiting(para):
            return 0.0

        return jax.lax.cond(
            para["c_x3"] ** 2 + para["c_y3"] ** 2 <= 1,
            fully_transiting,
            not_transiting,
            para,
        )

    def partially_transiting(X):
        para, xs, ys = X

        alphas = cartesian_intersection_to_parametric_angle(xs, ys, **para)
        alphas = jnp.where(xs != 999, alphas, 2 * jnp.pi)
        alphas = jnp.where(alphas < 0, alphas + 2 * jnp.pi, alphas)
        alphas = jnp.where(alphas > 2 * jnp.pi, alphas - 2 * jnp.pi, alphas)
        alphas = jnp.sort(alphas)

        test_ang = alphas[0] + (alphas[1] - alphas[0]) / 2
        test_ang = jnp.where(test_ang > 2 * jnp.pi, test_ang - 2 * jnp.pi, test_ang)

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
            solution_vectors = planet_solution_vec(
                alphas[0], alphas[1], g_coeffs, **para
            )
            planet_contribution = (
                jnp.matmul(solution_vectors, g_coeffs) * normalization_constant
            )
            return planet_contribution

        def testval_outside_star(_):
            leg1_solution_vec = planet_solution_vec(
                alphas[1], 2 * jnp.pi, g_coeffs, **para
            )
            leg1 = jnp.matmul(leg1_solution_vec, g_coeffs)
            leg2_solution_vec = planet_solution_vec(0.0, alphas[0], g_coeffs, **para)
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
            jnp.matmul(star_solution_vectors, g_coeffs) * normalization_constant
        )

        total_blocked = planet_contribution + star_contribution

        return total_blocked

    def transiting(X):
        indv_para, indv_two = X
        (
            xs,
            ys,
        ) = _single_intersection_points(**indv_two)

        on_limb = jnp.sum(xs) != 999 * 4

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
