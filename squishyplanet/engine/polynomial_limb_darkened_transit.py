import jax

jax.config.update("jax_enable_x64", True)
from functools import partial

import jax.numpy as jnp
import numpy as np

from squishyplanet.engine.kepler import kepler, skypos, t0_to_t_peri
from squishyplanet.engine.parametric_ellipse import (
    cartesian_intersection_to_parametric_angle,
    point_in_ellipse,
    poly_to_parametric,
)

# Fixed-grid quadrature for the planet solution-vector integrals, replacing the
# previous adaptive quadax.quadgk call. A single unrolled pass is faster to
# evaluate and differentiate than the old adaptive refinement loop. The integrand
# behaves differently in the two ways planet_solution_vec is called, so we use a
# different fixed rule for each:
#
#   * Partial-transit arcs (a, b are the parametric angles where the outline
#     crosses the stellar limb): the integrand has a square-root-type singularity
#     at BOTH endpoints, where 1 - x^2 - y^2 -> 0. We integrate on a cubic
#     "smoothstep" reparameterization of [a, b] with zero derivative at each
#     endpoint, which regularizes the sqrt singularity into a smooth integrand and
#     recovers spectral GL convergence (solution-vector value error ~1e-14 and
#     jacfwd relative error ~1e-10 worst case over LD orders 2-8 and a wide range
#     of geometries, vs the old adaptive scheme's ~1e-7 jacfwd error).
#   * Full transit (a=0, b=2*pi, planet fully inside the star): the integrand is
#     smooth and periodic with no endpoint singularity, but can develop a sharp
#     near-cusp mid-arc when the outline grazes the limb internally
#     (1 - x^2 - y^2 -> 0 away from the endpoints). The smoothstep clusters nodes
#     at the endpoints -- the wrong place -- and under-resolves that feature. Here
#     we use the periodic trapezoid rule, which is spectrally accurate for smooth
#     periodic integrands and resolves the interior near-cusp far more efficiently
#     than a plain GL rule.
#
# The two schemes meet at internal tangency (d_center = 1 - r, i.e. second/third
# contact), where the full-circle integrand is at its most grazing. Their
# independent errors there set the size of any flux discontinuity at the handoff.
# N_TRAP is chosen so that worst-case step is ~5e-12 -- below the genuine physical
# contact-point kink (~2e-11) that any transit model has -- so the scheme switch
# introduces no discontinuity beyond the unavoidable physics.
#
# GL nodes are open (never land on +/-1), so the smoothstep samples never hit the
# exact endpoints and we avoid sqrt(negative)/nan even if the quartic roots are
# slightly off the limb.
N_GL = 48  # smoothstep GL nodes for partial-transit arcs
N_TRAP = 256  # periodic trapezoid nodes for the full circle
_gl_nodes, _gl_weights = np.polynomial.legendre.leggauss(N_GL)
# smoothstep map phi(t) = (3 t - t^3) / 2 on [-1, 1] (phi(+/-1) = +/-1) and its
# derivative dphi(t) = (3 - 3 t^2) / 2 (dphi(+/-1) = 0), evaluated at the nodes.
_GL_PHI = jnp.asarray((3.0 * _gl_nodes - _gl_nodes**3) / 2.0)
_GL_WDPHI = jnp.asarray(_gl_weights * (3.0 - 3.0 * _gl_nodes**2) / 2.0)
# Equally-spaced nodes for the periodic trapezoid (full circle), offset by half a
# step to (k + 1/2) * 2*pi / N_TRAP. This is equally spectrally accurate for smooth
# periodic integrands but keeps nodes off the "nice" angles 0, pi/2, pi, ... so they
# don't sit exactly on the grazing tangent of an axis-aligned outline (where
# 1 - x^2 - y^2 = 0). The integrand is nan-safe there regardless; this just avoids
# sampling the cusp itself.
_TRAP_NODES = jnp.asarray((np.arange(N_TRAP) + 0.5) * (2.0 * np.pi / N_TRAP))
_TRAP_WEIGHT = 2.0 * jnp.pi / N_TRAP
# A full-circle call is detected by its span being (numerically) 2*pi.
_FULL_SPAN = 2.0 * jnp.pi - 1e-9


def _t4(
    rho_xx: jax.Array,
    rho_xy: jax.Array,
    rho_x0: jax.Array,
    rho_yy: jax.Array,
    rho_y0: jax.Array,
    rho_00: jax.Array,
) -> jax.Array:
    return -1 + rho_00 - rho_x0 + rho_xx


def _t3(
    rho_xx: jax.Array,
    rho_xy: jax.Array,
    rho_x0: jax.Array,
    rho_yy: jax.Array,
    rho_y0: jax.Array,
    rho_00: jax.Array,
) -> jax.Array:
    return -2 * rho_xy + 2 * rho_y0


def _t2(
    rho_xx: jax.Array,
    rho_xy: jax.Array,
    rho_x0: jax.Array,
    rho_yy: jax.Array,
    rho_y0: jax.Array,
    rho_00: jax.Array,
) -> jax.Array:
    return -2 + 2 * rho_00 - 2 * rho_xx + 4 * rho_yy


def _t1(
    rho_xx: jax.Array,
    rho_xy: jax.Array,
    rho_x0: jax.Array,
    rho_yy: jax.Array,
    rho_y0: jax.Array,
    rho_00: jax.Array,
) -> jax.Array:
    return 2 * rho_xy + 2 * rho_y0


def _t0(
    rho_xx: jax.Array,
    rho_xy: jax.Array,
    rho_x0: jax.Array,
    rho_yy: jax.Array,
    rho_y0: jax.Array,
    rho_00: jax.Array,
) -> jax.Array:
    return -1 + rho_00 + rho_x0 + rho_xx


@jax.jit
def _single_intersection_points(
    rho_xx: jax.Array,
    rho_xy: jax.Array,
    rho_x0: jax.Array,
    rho_yy: jax.Array,
    rho_y0: jax.Array,
    rho_00: jax.Array,
    **kwargs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
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
def parameterize_2d_helper(
    projected_r: jax.Array,
    projected_f: jax.Array,
    projected_theta: jax.Array,
    xc: jax.Array,
    yc: jax.Array,
) -> tuple[dict, dict]:
    """Convert from the alternative sky-projected parameterization to the same format
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

    # two alternative takes cooked up during the Fortran implementation-
    # check later for numerical implications

    # projected_r_sq = projected_r * projected_r
    # projected_r2_sq = projected_r2 * projected_r2
    # cos_t_sq = cos_t * cos_t
    # sin_t_sq = sin_t * sin_t
    # xc_sq = xc * xc
    # yc_sq = yc * yc

    # two = {
    #     "rho_xx": cos_t_sq / projected_r_sq + sin_t_sq / projected_r2_sq,
    #     "rho_xy": (2 * cos_t * sin_t) / projected_r_sq
    #     - (2 * cos_t * sin_t) / projected_r2_sq,
    #     "rho_x0": (
    #         ((-2 * cos_t_sq * xc) - (2 * cos_t * yc * sin_t)) / projected_r_sq
    #         + ((2 * cos_t * yc * sin_t) - (2 * xc * sin_t_sq)) / projected_r2_sq
    #     ),
    #     "rho_yy": cos_t_sq / projected_r2_sq + sin_t_sq / projected_r_sq,
    #     "rho_y0": (
    #         (- (2 * cos_t * xc * sin_t) - (2 * yc * sin_t_sq)) / projected_r_sq +
    #         ((-2 * cos_t_sq * yc) + (2 * cos_t * xc * sin_t)) / projected_r2_sq
    #     ),
    #     "rho_00": (
    #         ((cos_t_sq * xc_sq) + (2 * cos_t * xc * yc * sin_t) + (yc_sq * sin_t_sq)) / projected_r_sq +
    #         ((cos_t_sq * yc_sq) - (2 * cos_t * xc * yc * sin_t) + (xc_sq * sin_t_sq)) / projected_r2_sq
    #     )
    # }
    # two = {}
    # two["rho_xx"] = cos_t_sq / projected_r_sq + sin_t_sq / projected_r2_sq

    # tmp1 = 2.0 * cos_t * sin_t
    # two["rho_xy"] = tmp1 / projected_r_sq - tmp1 / projected_r2_sq

    # tmp1 = 2.0 * cos_t * yc * sin_t
    # two["rho_x0"] = ((-2.0 * cos_t_sq * xc) - tmp1) / projected_r_sq + (
    #     tmp1 - (2.0 * xc * sin_t_sq)
    # ) / projected_r2_sq

    # two["rho_yy"] = cos_t_sq / projected_r2_sq + sin_t_sq / projected_r_sq

    # tmp1 = 2.0 * cos_t * xc * sin_t
    # two["rho_y0"] = ((-2.0 * cos_t_sq * yc) + (tmp1)) / projected_r2_sq - (
    #     (2.0 * cos_t * xc * sin_t) + (2.0 * yc * sin_t_sq)
    # ) / projected_r_sq

    # tmp1 = 2.0 * cos_t * xc * yc * sin_t
    # two["rho_00"] = (
    #     (cos_t_sq * xc_sq) + (tmp1) + (yc_sq * sin_t_sq)
    # ) / projected_r_sq + (
    #     (cos_t_sq * yc_sq) - (tmp1) + (xc_sq * sin_t_sq)
    # ) / projected_r2_sq

    para = poly_to_parametric(**two)

    return two, para


@jax.jit
def planet_solution_vec(
    a: jax.Array,
    b: jax.Array,
    g_coeffs: jax.Array,
    c_x1: jax.Array,
    c_x2: jax.Array,
    c_x3: jax.Array,
    c_y1: jax.Array,
    c_y2: jax.Array,
    c_y3: jax.Array,
) -> jax.Array:
    """Compute the "solution vector" for a 1D path across the star that lies on the outline
    of the planet.

    This computes Eq. 21 of `Agol, Luger, and Foreman-Mackey 2020
    <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_. But, instead of
    doing it analytically, this numerically solves the required integrals with a
    fixed-grid Gauss-Legendre rule (see ``N_GL`` and the module-level notes). For terms
    s_2 and higher, this is straightforward to do based on
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

    # The s0, s1, and higher-order sn integrals are evaluated together as a single
    # vector-valued integrand sharing one fixed Gauss-Legendre node set (instead of
    # three independent integrations over the same interval). The lax.scan over
    # polynomial orders becomes a vectorized power.
    ns = jnp.arange(g_coeffs.shape[0])[2:]

    def vec_integrand(s: jax.Array) -> jax.Array:
        cs = jnp.cos(s)
        sn = jnp.sin(s)
        x = cs * c_x1 + sn * c_x2 + c_x3
        y = cs * c_y1 + sn * c_y2 + c_y3
        dy_da = -sn * c_y1 + cs * c_y2
        # Guard the grazing tangent. When the outline just touches the stellar limb
        # (fully-inside case), 1 - x^2 - y^2 -> 0 at one angle, and a fixed quadrature
        # node can land on or just past it. Two failure modes follow: (1) rounding of
        # x^2 + y^2 can make 1 - x^2 - y^2 slightly negative (platform-dependent),
        # giving sqrt(neg)=nan and (neg)**(n/2)=nan; (2) at exactly 1 - x^2 - y^2 = 0,
        # sqrt and arctan(x/root) have infinite derivatives, giving a nan *gradient*
        # even when the value is fine. The true outline has 1 - x^2 - y^2 >= 0 here, so
        # we use a nan-safe formulation: clamp, a guarded sqrt, and arctan2 (which
        # equals arctan(x/root) for root > 0 but stays finite in value and gradient at
        # root == 0). This is a no-op wherever 1 - x^2 - y^2 > 0.
        om = 1.0 - x**2 - y**2
        pos = om > 0.0
        one_minus = jnp.where(pos, om, 0.0)
        root = jnp.where(pos, jnp.sqrt(jnp.where(pos, om, 1.0)), 0.0)

        v0 = x * dy_da
        v1 = (
            dy_da * (jnp.pi + 6 * x * root - 6 * jnp.arctan2(x, root) * (-1 + y**2))
        ) / 12.0

        common = -(
            c_x3 * (sn * c_y1 - cs * c_y2)
            + c_x2 * (c_y1 + cs * c_y3)
            - c_x1 * (c_y2 + sn * c_y3)
        )
        vn = one_minus ** (ns / 2.0) * common
        return jnp.concatenate((jnp.array([v0, v1]), vn))

    # Full-circle calls use the periodic trapezoid; partial arcs use the smoothstep
    # reparameterization of [a, b] that regularizes the sqrt endpoint singularity:
    #   s = mid + half * phi(t),  integral = half * sum_k w_k dphi(t_k) f(s_k).
    def full_circle(_: None) -> jax.Array:
        vals = jax.vmap(vec_integrand)(_TRAP_NODES)
        return _TRAP_WEIGHT * jnp.sum(vals, axis=0)

    def partial_arc(_: None) -> jax.Array:
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        vals = jax.vmap(vec_integrand)(mid + half * _GL_PHI)
        return half * jnp.sum(_GL_WDPHI[:, None] * vals, axis=0)

    return jax.lax.cond((b - a) >= _FULL_SPAN, full_circle, partial_arc, None)


# Closed-form antiderivatives for the star-arc integrands (see star_solution_vec).
_HALF_PI = jnp.pi / 2.0
_THREE_HALF_PI = 3.0 * jnp.pi / 2.0
_TWO_PI = 2.0 * jnp.pi


def _F0(t: jax.Array) -> jax.Array:
    """Antiderivative of ``cos^2(t)``: ``t/2 + sin(2t)/4``."""
    return t / 2.0 + jnp.sin(2.0 * t) / 4.0


def _F_outer(t: jax.Array) -> jax.Array:
    """Antiderivative of the ``s1`` outer piece, ``(pi/12)(4 sin t - sin^3 t)``."""
    s = jnp.sin(t)
    return (jnp.pi / 12.0) * (4.0 * s - s**3)


def _F_inner(t: jax.Array) -> jax.Array:
    """Antiderivative of the ``s1`` inner piece, ``(pi/12)(-2 sin t + sin^3 t)``."""
    s = jnp.sin(t)
    return (jnp.pi / 12.0) * (-2.0 * s + s**3)


def _s0_definite(lo: jax.Array, hi: jax.Array) -> jax.Array:
    """Closed-form ``integral_{lo}^{hi} cos^2(t) dt``."""
    return _F0(hi) - _F0(lo)


def _s1_definite(lo: jax.Array, hi: jax.Array) -> jax.Array:
    """Closed-form ``integral_{lo}^{hi} s1_integrand(t) dt`` for ``0 <= lo <= hi <= 2 pi``.

    Splits the interval at ``pi/2`` and ``3 pi/2`` by clamping the limits into each
    region; the outer antiderivative is used on ``[0, pi/2]`` and ``[3 pi/2, 2 pi]``,
    the inner antiderivative on ``[pi/2, 3 pi/2]``.
    """
    a1 = jnp.clip(lo, 0.0, _HALF_PI)
    b1 = jnp.clip(hi, 0.0, _HALF_PI)
    c1 = _F_outer(b1) - _F_outer(a1)

    a2 = jnp.clip(lo, _HALF_PI, _THREE_HALF_PI)
    b2 = jnp.clip(hi, _HALF_PI, _THREE_HALF_PI)
    c2 = _F_inner(b2) - _F_inner(a2)

    a3 = jnp.clip(lo, _THREE_HALF_PI, _TWO_PI)
    b3 = jnp.clip(hi, _THREE_HALF_PI, _TWO_PI)
    c3 = _F_outer(b3) - _F_outer(a3)

    return c1 + c2 + c3


@jax.jit
def star_arc_solution_vec(
    theta_lo: jax.Array,
    theta_hi: jax.Array,
    g_coeffs: jax.Array,
) -> jax.Array:
    """Compute the "solution vector" for a 1D path that lies on the edge of the star.

    This is equivalent to :func:`planet_solution_vec`, but instead of integrating over
    paths that lie on the planet's outline, we integrate over paths that lie on the edge
    of the star. As pointed out in the paragraph following Eq. 69 in `Agol, Luger, and
    Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_,
    the contribution of all terms higher than :math:`G_1` will be zero in this case
    since we have limited ourselves to :math:`z=0` by remaining on the star's boundary.
    This simplifies things somewhat: the dot product of the parametric form of the
    star's outline with the :math:`G_0` and :math:`G_1` terms written out in
    :func:`planet_solution_vec` reduces to elementary trigonometric polynomials, so we
    evaluate the required integrals in closed form via their exact antiderivatives
    rather than numerically.

    Unlike its predecessor ``star_solution_vec``, this function takes the polar angles
    of the arc endpoints on the star directly (not the parametric angles of another
    curve's outline), and it makes no assumption about which of the arcs bounded by two
    intersection points is wanted: the caller chooses by passing the endpoints in
    counterclockwise order. This matters once more than one curve can cut the stellar
    limb into several arcs.

    Args:
        theta_lo (float):
            The starting polar angle of the arc on the star, in ``[0, 2 pi]``.
        theta_hi (float):
            The ending polar angle of the arc on the star. Must satisfy
            ``theta_lo <= theta_hi <= 2 pi``; the arc is traversed counterclockwise
            from ``theta_lo`` to ``theta_hi``. Arcs crossing the ``0 = 2 pi`` seam
            should be passed as two separate sub-arcs.
        g_coeffs (Array):
            The system-specific limb darkening coefficients in the Green's basis.
            Computed by multiplying the u coefficients with the change of basis matrix
            from :func:`greens_basis_transform.generate_change_of_basis_matrix`.

    Returns:
        Array:
            The solution vector for the path along the star's edge. The shape will
            match that of the input ``g_coeffs``.

    """
    solution_vec = jnp.zeros(g_coeffs.shape[0])
    solution_vec = solution_vec.at[0].set(_s0_definite(theta_lo, theta_hi))
    solution_vec = solution_vec.at[1].set(_s1_definite(theta_lo, theta_hi))
    return solution_vec


def _rot(angle: jax.Array, axis: str) -> jax.Array:
    """Batched rotation matrix about ``axis`` ('x', 'y', or 'z').

    Args:
        angle (Array): Rotation angle(s) [radian], any leading shape ``(...)``.
        axis (str): One of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        Array: Rotation matrices of shape ``(..., 3, 3)``.
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    o, z = jnp.ones_like(c), jnp.zeros_like(c)
    if axis == "x":
        rows = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "y":
        rows = [[c, z, s], [z, o, z], [-s, z, c]]
    elif axis == "z":
        rows = [[c, -s, z], [s, c, z], [z, z, o]]
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return jnp.stack([jnp.stack(row, -1) for row in rows], -2)


def outline_prelude(state: dict) -> tuple[dict, dict]:
    """Direct construction of the planet's projected outline from orbital elements.

    Drop-in replacement for the ``planet_3d_coeffs -> planet_2d_coeffs ->
    poly_to_parametric`` chain (the full 3D-parameterization branch of
    :func:`lightcurve`). Instead of expanding the implicit polynomials, it builds the
    central 3D quadric ``M = R diag(d) R^T`` directly, eliminates ``z`` via a 2x2 Schur
    complement to get the projected conic, and uses the planet's sky position
    ``skypos(...)[:2]`` as the outline center exactly. This avoids the near-cancellation
    divisions of the old chain and is machine-precision accurate (vs the chain's ~1e-9
    worst case) as well as slightly faster.

    Expects ``state["f"]`` to already hold the per-timestep true anomalies. Applies the
    tidally-locked precession override internally (``prec = where(tidally_locked, f,
    prec)``).

    Args:
        state (dict):
            An :func:`OblateSystem` ``state`` dictionary. Uses the keys ``a, e, f,
            Omega, i, omega, r, obliq, prec, f1, f2, tidally_locked``.

    Returns:
        tuple:
            A tuple of two dictionaries. The first contains the implicit-conic
            coefficients ``rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00`` of the
            outline (matching :func:`planet_2d.planet_2d_coeffs`). The second contains
            the parametric coefficients ``c_x1, c_x2, c_x3, c_y1, c_y2, c_y3`` with
            ``x = c_x1 cos(alpha) + c_x2 sin(alpha) + c_x3`` (and similarly for y),
            matching :func:`parametric_ellipse.poly_to_parametric`. Every leaf is
            broadcast to the time axis.
    """
    prec = jnp.where(state["tidally_locked"], state["f"], state["prec"])

    # central 3D quadric M = R diag(d) R^T (depends only on orientation + shape)
    R = (
        _rot(state["Omega"], "z")
        @ _rot(state["i"], "x")
        @ _rot(state["omega"] + prec, "z")
        @ _rot(state["obliq"], "y")
    )
    r = jnp.asarray(state["r"]).ravel()[0]
    f1 = jnp.asarray(state["f1"]).ravel()[0]
    f2 = jnp.asarray(state["f2"]).ravel()[0]
    d = jnp.stack(
        [1 / r**2, 1 / (r * (1 - f2)) ** 2, 1 / (r * (1 - f1)) ** 2],
        -1,
    )
    M = jnp.einsum("...ij,...j,...kj->...ik", R, d, R)

    # 2x2 outline conic via Schur complement eliminating z -> centered conic
    #   A x'^2 + B x'y' + C y'^2 = 1
    m22 = M[..., 2, 2]
    col = M[..., :2, 2]
    row = M[..., 2, :2]
    Amat = (
        M[..., :2, :2] - (col[..., :, None] * row[..., None, :]) / m22[..., None, None]
    )
    sxx = Amat[..., 0, 0]  # A
    sxy = 2 * Amat[..., 0, 1]  # B
    syy = Amat[..., 1, 1]  # C

    # ellipse center is exactly the projected sky position
    pos = skypos(
        state["a"], state["e"], state["f"], state["Omega"], state["i"], state["omega"]
    )
    xc = pos[0]
    yc = pos[1]

    # parametric ellipse: identical convention to poly_to_parametric_helper
    theta = jnp.where(
        sxx - syy != 0.0,
        0.5 * jnp.arctan2(sxy, (sxx - syy)) + jnp.pi / 2,
        0.0,
    )
    theta = jnp.where(theta < 0.0, theta + jnp.pi, theta)
    cosa = jnp.cos(theta)
    sina = jnp.sin(theta)
    aa = sxx * cosa**2 + sxy * cosa * sina + syy * sina**2
    bb = sxx * sina**2 - sxy * cosa * sina + syy * cosa**2
    r1 = 1 / jnp.sqrt(aa)
    r2 = 1 / jnp.sqrt(bb)

    # xc/yc always vary with the true anomaly (length n_times); the orientation-only
    # quantities are scalar when prec is fixed. Broadcast them to the time axis so every
    # leaf shares a leading dimension for the downstream lax.scan (this subsumes the
    # rho_xx/rho_xy/rho_yy broadcast block the old 3D branch needed below).
    ones = jnp.ones_like(xc)

    para = {
        "c_x1": ones * (r1 * cosa),
        "c_x2": ones * (-r2 * sina),
        "c_x3": xc,
        "c_y1": ones * (r1 * sina),
        "c_y2": ones * (r2 * cosa),
        "c_y3": yc,
    }

    two = {
        "rho_xx": ones * sxx,
        "rho_xy": ones * sxy,
        "rho_x0": -2 * sxx * xc - sxy * yc,
        "rho_yy": ones * syy,
        "rho_y0": -2 * syy * yc - sxy * xc,
        "rho_00": sxx * xc**2 + sxy * xc * yc + syy * yc**2,
    }

    return two, para


def ellipse_bound(
    c_x1: jax.Array,
    c_x2: jax.Array,
    c_x3: jax.Array,
    c_y1: jax.Array,
    c_y2: jax.Array,
    c_y3: jax.Array,
) -> jax.Array:
    """Decide whether a step straddles the star edge (and so needs the quartic solve).

    The outline center is ``(c_x3, c_y3)`` and a safe upper bound on the distance of any
    outline point from the center is ``bound = sqrt(c_x1^2 + c_x2^2 + c_y1^2 + c_y2^2)``
    (it equals ``sqrt(r1^2 + r2^2) >= max(r1, r2)``). With center distance
    ``d = sqrt(c_x3^2 + c_y3^2)``: ``d + bound < 1`` is provably fully inside,
    ``d - bound > 1`` provably fully outside; otherwise the step straddles and the
    quartic must be solved. A conservative bound is safe -- a straddling-classified step
    with no real root is harmless (the center-inside test resolves it).

    Args:
        c_x1 (float): Parametric outline coefficient (x, cos term).
        c_x2 (float): Parametric outline coefficient (x, sin term).
        c_x3 (float): Parametric outline coefficient (x, center).
        c_y1 (float): Parametric outline coefficient (y, cos term).
        c_y2 (float): Parametric outline coefficient (y, sin term).
        c_y3 (float): Parametric outline coefficient (y, center).

    Returns:
        bool: ``True`` if straddling (solve the quartic), ``False`` if provably fully
        inside or fully outside.
    """
    bound = jnp.sqrt(c_x1**2 + c_x2**2 + c_y1**2 + c_y2**2)
    d = jnp.sqrt(c_x3**2 + c_y3**2)
    return (d + bound >= 1.0) & (d - bound <= 1.0)


def _arcs_from_angles(angles: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split ``[0, 2 pi]`` into consecutive arcs at the given angles.

    This is the "bookend" scheme: the (sentinel-padded) angles are sorted and the fixed
    endpoints 0 and 2 pi are prepended/appended, so ``n`` input slots always produce
    ``n + 1`` arc slots of static shape. Callers should map sentinel entries (no
    intersection) to 2 pi beforehand; those slots become zero-length arcs
    ``[2 pi, 2 pi]`` that integrate to zero. A boundary crossing the parameterization
    seam is represented as two sub-arcs, ``[a_last, 2 pi]`` and ``[0, a_first]``, whose
    contributions add, so no arc ever wraps. With no genuine angles at all, the single
    arc ``[0, 2 pi]`` survives and its midpoint doubles as a containment test point for
    the whole curve.

    Args:
        angles (Array): Angles in ``[0, 2 pi]``, sentinel slots mapped to 2 pi.

    Returns:
        Tuple:
            Three arrays of length ``len(angles) + 1``: the arc lower bounds, upper
            bounds, and midpoints.

    """
    angles = jnp.sort(angles)
    lo = jnp.concatenate((jnp.zeros(1), angles))
    hi = jnp.concatenate((angles, jnp.full(1, 2 * jnp.pi)))
    mid = 0.5 * (lo + hi)
    return lo, hi, mid


def ellipse_star_term(para: dict, two: dict, g_coeffs: jax.Array) -> jax.Array:
    """The Green's-theorem boundary integral of the blocked flux over one ellipse.

    Computes :math:`F(E \\cap S)`, the flux blocked by the intersection of a single
    opaque ellipse :math:`E` (a planet outline or a ring edge) with the stellar disk
    :math:`S`, dotted with the Green's-basis limb-darkening coefficients. The caller is
    responsible for multiplying by the overall normalization constant.

    Rather than assuming a fixed number of ellipse-star intersections, this takes an
    "exploratory" approach that handles 0 through 4 crossings (and containment in
    either direction) with one code path: find all crossings, split both the ellipse
    outline and the stellar limb into arcs at those crossings, keep each arc if its
    midpoint lies inside the other region (its inside/outside status is constant along
    the arc, since arcs are split at every crossing), and sum the kept arcs'
    contributions. Because the boundary of the intersection of convex regions is
    exactly the set of arcs of each curve that lie inside the other, and both curves
    are traversed counterclockwise, Green's theorem makes the summation order
    irrelevant -- no case enumeration or boundary stitching is needed.

    Args:
        para (dict):
            The parametric coefficients ``c_x1 ... c_y3`` of the ellipse outline for a
            single timestep. Must trace the outline counterclockwise (true of
            everything produced by :func:`poly_to_parametric`,
            :func:`outline_prelude`, and :func:`rings.ring_para_coeffs`).
        two (dict):
            The implicit-conic coefficients ``rho_xx ... rho_00`` of the same ellipse.
        g_coeffs (Array):
            The system-specific limb darkening coefficients in the Green's basis.

    Returns:
        Array:
            Scalar: the blocked flux from this ellipse, before normalization.

    """

    def straddling_case(X: tuple) -> jax.Array:
        para, two = X
        xs, ys = _single_intersection_points(**two)

        # arcs of the ellipse outline, split at the crossings; keep those inside the
        # star (zero-length sentinel arcs are skipped to save their quadrature calls)
        alphas = cartesian_intersection_to_parametric_angle(xs, ys, **para)
        alphas = jnp.where(xs != 999, alphas, 2 * jnp.pi)
        alphas = jnp.where(alphas < 0, alphas + 2 * jnp.pi, alphas)
        alphas = jnp.where(alphas > 2 * jnp.pi, alphas - 2 * jnp.pi, alphas)
        lo, hi, mid = _arcs_from_angles(alphas)
        mx = para["c_x1"] * jnp.cos(mid) + para["c_x2"] * jnp.sin(mid) + para["c_x3"]
        my = para["c_y1"] * jnp.cos(mid) + para["c_y2"] * jnp.sin(mid) + para["c_y3"]
        keep = (mx**2 + my**2 < 1.0) & (hi > lo)

        def ellipse_arc(carry: jax.Array, arc: tuple) -> tuple[jax.Array, None]:
            arc_lo, arc_hi, arc_keep = arc
            val = jax.lax.cond(
                arc_keep,
                lambda _: jnp.matmul(
                    planet_solution_vec(arc_lo, arc_hi, g_coeffs, **para), g_coeffs
                ),
                lambda _: 0.0,
                None,
            )
            return carry + val, None

        ellipse_contribution = jax.lax.scan(ellipse_arc, 0.0, (lo, hi, keep))[0]

        # arcs of the stellar limb, split at the same crossings (now in polar angle);
        # keep those inside the ellipse. These are closed-form and cheap, so no
        # cond-gating is needed
        thetas = jnp.where(xs != 999, jnp.arctan2(ys, xs), 2 * jnp.pi)
        thetas = jnp.where(thetas < 0, thetas + 2 * jnp.pi, thetas)
        star_lo, star_hi, star_mid = _arcs_from_angles(thetas)
        keep_star = point_in_ellipse(jnp.cos(star_mid), jnp.sin(star_mid), **para) & (
            star_hi > star_lo
        )
        s0 = _s0_definite(star_lo, star_hi)
        s1 = _s1_definite(star_lo, star_hi)
        star_contribution = jnp.sum(
            jnp.where(keep_star, g_coeffs[0] * s0 + g_coeffs[1] * s1, 0.0)
        )

        return ellipse_contribution + star_contribution

    def not_straddling_case(X: tuple) -> jax.Array:
        para, _ = X

        def fully_transiting(para: dict) -> jax.Array:
            return jnp.matmul(
                planet_solution_vec(a=0.0, b=2 * jnp.pi, g_coeffs=g_coeffs, **para),
                g_coeffs,
            )

        def not_transiting(para: dict) -> float:
            return 0.0

        return jax.lax.cond(
            para["c_x3"] ** 2 + para["c_y3"] ** 2 <= 1,
            fully_transiting,
            not_transiting,
            para,
        )

    # conservative bounding pre-test: only outlines that straddle the star edge need
    # the quartic solve. Note an ellipse that contains the whole star always classifies
    # as straddling, and is handled by the crossing-free containment logic above.
    return jax.lax.cond(
        ellipse_bound(**para), straddling_case, not_straddling_case, (para, two)
    )


def _lightcurve_setup(state: dict, parameterize_with_projected_ellipse: bool) -> dict:
    """Shared vectorized setup for the transit drivers.

    Everything in :func:`lightcurve` that happens before the per-timestep
    ``jax.lax.scan``: solving Kepler's equation for the true anomalies, converting the
    limb-darkening ``u`` coefficients to the Green's basis, and building the
    per-timestep implicit and parametric outline coefficients. Split out so that a
    ringed system can compute this once and reuse it across the planet and ring-edge
    integrations.

    Note that this also stores the true anomalies in ``state["f"]`` (and
    ``state["t_peri"]`` if ``t0`` was provided) as a side effect, matching the previous
    inline behavior.

    Args:
        state (dict):
            A dictionary containing all of the keys that are included in an
            :func:`OblateSystem` ``state`` attribute.
        parameterize_with_projected_ellipse (bool):
            See :func:`lightcurve`.

    Returns:
        dict:
            Keys: ``fluxes`` (all ones, shape of ``times``), ``g_coeffs``,
            ``normalization_constant``, ``positions`` (from :func:`kepler.skypos`),
            ``two`` and ``para`` (per-timestep outline coefficient dicts), and
            ``largest_r`` (a bound on the planet's projected radius for the
            possibly-in-transit mask).

    """
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

        # if prec isn't the same length as f, parameterize_2d_helper left the
        # orientation-only quadratic terms as scalars while the centers are length-n.
        # Broadcast them to match for the downstream lax.scan.
        if state["prec"].shape != true_anomalies.shape:
            two["rho_xx"] = jnp.ones_like(state["f"]) * two["rho_xx"]
            two["rho_xy"] = jnp.ones_like(state["f"]) * two["rho_xy"]
            two["rho_yy"] = jnp.ones_like(state["f"]) * two["rho_yy"]

    else:
        # direct orbital-elements -> outline construction (M = R D R^T + Schur),
        # replacing the planet_3d_coeffs -> planet_2d_coeffs -> poly_to_parametric
        # chain. It applies the tidally-locked prec override and broadcasts every leaf
        # to the time axis internally.
        two, para = outline_prelude(state)

        largest_r = state["r"]

    return {
        "fluxes": fluxes,
        "g_coeffs": g_coeffs,
        "normalization_constant": normalization_constant,
        "positions": positions,
        "two": two,
        "para": para,
        "largest_r": largest_r,
    }


@partial(jax.jit, static_argnames=("parameterize_with_projected_ellipse",))
def lightcurve(state: dict, parameterize_with_projected_ellipse: bool) -> jax.Array:
    """The main function for computing a transit light curve.

    This function will return a 1-D array representing the flux received from the star,
    where each entry corresponds to a time in the input `state` dictionary. It first
    transforms the `state` into the implicit 3D surface of the planet, the implicit 2D
    sky-projected outline of the planet, and a parametric form of that outline for each
    time step. These are vectorized operations that are computed simultaneously across
    all times. It then solves for the intersection points of the planet and star, and
    if the planet is either partially or fully transiting, numerically solves the
    required 1D integrals that leverage Green's Theorem to compute the blocked flux. The
    flux-blocking calculations are done sequentially for each timestep using
    ``jax.lax.scan``, which seemed to be more efficient than vectorizing again while
    switching between branches with something like ``jax.lax.cond``. Keep these different
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
    setup = _lightcurve_setup(state, parameterize_with_projected_ellipse)
    g_coeffs = setup["g_coeffs"]
    normalization_constant = setup["normalization_constant"]
    positions = setup["positions"]

    possibly_in_transit = (
        positions[0, :] ** 2 + positions[1, :] ** 2
        <= (1.0 + setup["largest_r"] * 1.1) ** 2
    ) * (positions[2, :] > 0)

    def transiting(X: tuple) -> jax.Array:
        indv_para, indv_two = X
        return ellipse_star_term(indv_para, indv_two, g_coeffs) * normalization_constant

    def not_transiting(X: tuple) -> float:
        return 0.0

    def scan_func(carry: None, scan_over: tuple) -> tuple[None, jax.Array]:
        indv_para, indv_two, mask = scan_over
        return None, jax.lax.cond(
            mask, transiting, not_transiting, (indv_para, indv_two)
        )

    transit_fluxes = jax.lax.scan(
        scan_func, None, (setup["para"], setup["two"], possibly_in_transit), None
    )[1]

    return setup["fluxes"] - transit_fluxes
