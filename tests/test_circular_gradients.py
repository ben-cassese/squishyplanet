"""Circular projections must retain derivatives toward noncircular shapes."""

from itertools import pairwise

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from squishyplanet import OblateSystem
from squishyplanet.engine.parametric_ellipse import parametric_to_poly
from squishyplanet.engine.polynomial_limb_darkened_transit import (
    _lightcurve_setup,
    outline_prelude,
    parameterize_2d_helper,
)


def _system(projected: bool, locked: bool = False) -> OblateSystem:
    return OblateSystem(
        times=jnp.array([-0.06, -0.0564, -0.03, 0.0, 0.03, 0.0564, 0.06]),
        t_peri=-0.75,
        period=3.0,
        a=8.0,
        e=0.0,
        i=float(np.arccos(0.3 / 8.0)),
        omega=0.0,
        r=0.1,
        f1=0.0,
        f2=0.0,
        obliq=0.4,
        prec=0.3,
        tidally_locked=locked,
        ld_u_coeffs=jnp.array([0.3, 0.2]),
        parameterize_with_projected_ellipse=projected,
        projected_effective_r=0.1,
        projected_f=0.0,
        projected_theta=0.7,
    )


def _circular_derivative(xc: float, yc: float, dq: np.ndarray) -> float:
    """Integrate the first variation of blocked flux, independently of JAX.

    For x^T Q x = 1 and Q = I/r^2, normal boundary velocity is
    -r^3/2 * n^T (dQ/dp) n. Only planet boundary inside the star contributes.
    """
    r = 0.1
    d = np.hypot(xc, yc)
    phi = np.arctan2(yc, xc)
    q = (1 - d**2 - r**2) / (2 * r * d)
    bounds = [0.0, 2 * np.pi]
    if abs(q) < 1:
        bounds.extend(np.mod([phi - np.arccos(q), phi + np.arccos(q)], 2 * np.pi))
    bounds.sort()

    def integrand(alpha: float) -> float:
        n = np.array([np.cos(alpha), np.sin(alpha)])
        x, y = np.array([xc, yc]) + r * n
        mu2 = 1 - x**2 - y**2
        if mu2 < 0:
            return 0.0
        mu = np.sqrt(mu2)
        intensity = 1 - 0.3 * (1 - mu) - 0.2 * (1 - mu) ** 2
        return intensity * (n @ dq @ n)

    integral = sum(
        quad(integrand, lo, hi, epsabs=1e-11, epsrel=1e-11)[0]
        for lo, hi in pairwise(bounds)
    )
    return r**4 / 2 * integral / (np.pi * (1 - 0.3 / 3 - 0.2 / 6))


@pytest.mark.parametrize("theta", [0.0, 0.7, np.pi / 4, 3 * np.pi / 4])
def test_projected_circular_flux_gradient(theta: float) -> None:
    system = _system(True)
    flux = jax.jit(
        lambda f: system.lightcurve({"projected_f": f, "projected_theta": theta})
    )
    derivative = jax.jit(jax.jacrev(flux))
    setup = _lightcurve_setup(dict(system.state), True)
    dq = 100 * np.array(
        [
            [-np.cos(2 * theta), -np.sin(2 * theta)],
            [-np.sin(2 * theta), np.cos(2 * theta)],
        ]
    )
    expected = np.array(
        [
            _circular_derivative(float(x), float(y), dq)
            for x, y in zip(*np.asarray(setup["positions"])[:2], strict=True)
        ]
    )
    np.testing.assert_allclose(derivative(0.0), expected, atol=1e-10, rtol=1e-7)
    np.testing.assert_allclose(derivative(1e-8), expected, atol=1e-10, rtol=1e-6)
    # Second-order forward differences stay within nonnegative flattening.
    for h in [1e-3, 1e-4]:
        fd = (-3 * flux(0.0) + 4 * flux(h) - flux(2 * h)) / (2 * h)
        np.testing.assert_allclose(fd, expected, atol=2e-8, rtol=1e-4)


@pytest.mark.parametrize(
    "locked,pole_on", [(False, False), (True, False), (False, True)]
)
def test_circular_3d_flux_gradients(locked: bool, pole_on: bool) -> None:
    system = _system(False, locked)
    keys = ["f1", "f2", "obliq", "prec"]
    p = (
        jnp.array([0.2, 0.0, np.arccos(0.3 / 8.0), np.pi / 2])
        if pole_on
        else jnp.array([0.0, 0.0, 0.4, 0.3])
    )
    flux = jax.jit(lambda p: system.lightcurve(dict(zip(keys, p, strict=True))))
    got = np.asarray(jax.jacrev(flux)(p))
    # Setup supplies the actual anomalies, including their time dependence.
    state = dict(system.state)
    setup = _lightcurve_setup(state, False)

    def conic(p: jax.Array) -> jax.Array:
        two, _ = outline_prelude(dict(state, **dict(zip(keys, p, strict=True))))
        return jnp.stack([two["rho_xx"], two["rho_xy"] / 2, two["rho_yy"]], axis=-1)

    dq = np.asarray(jax.jacfwd(conic)(p))
    expected = np.empty_like(got)
    for t, (x, y) in enumerate(np.asarray(setup["positions"])[:2].T):
        for k in range(4):
            expected[t, k] = _circular_derivative(
                x, y, np.array([[dq[t, 0, k], dq[t, 1, k]], [dq[t, 1, k], dq[t, 2, k]]])
            )
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=1e-7)
    for h in [1e-4, 1e-5]:
        fd = np.stack(
            [
                np.asarray(
                    (-3 * flux(p) + 4 * flux(p + h * v) - flux(p + 2 * h * v)) / (2 * h)
                )
                for v in jnp.eye(4)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(got, fd, atol=2e-8, rtol=1e-4)


@pytest.mark.parametrize("pole_on", [False, True])
def test_3d_outline_retains_conic_derivatives(pole_on: bool) -> None:
    # With i=0 and obliq=0, a body flattened along z is viewed pole-on.
    state = {
        "a": 8.0,
        "e": 0.0,
        "f": jnp.array([0.3]),
        "Omega": 0.0,
        "i": 0.0,
        "omega": 0.0,
        "r": 0.1,
        "obliq": 0.0,
        "prec": 0.7,
        "f1": 0.0,
        "f2": 0.0,
        "tidally_locked": False,
    }
    p = jnp.array([0.2 if pole_on else 0.0, 0.0, 0.0])

    def pair(p: jax.Array) -> tuple[jax.Array, jax.Array]:
        two, para = outline_prelude(dict(state, f1=p[0], f2=p[1], obliq=p[2]))
        reconstructed = parametric_to_poly(**para)
        keys = ["rho_xx", "rho_xy", "rho_yy"]
        return tuple(jnp.stack([c[k] for k in keys]) for c in [two, reconstructed])

    for offset in [0.0, 1e-7]:
        q = p.at[1].add(offset)
        expected, got = pair(q)
        np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-12)
        expected, got = jax.jacfwd(pair)(q)
        np.testing.assert_allclose(got, expected, atol=1e-9, rtol=1e-11)


def test_projected_outline_retains_cross_term_derivative() -> None:
    def pair(f: jax.Array) -> tuple[jax.Array, jax.Array]:
        two, para = parameterize_2d_helper(
            0.1 / jnp.sqrt(1 - f), f, 0.7, jnp.array(0.0), jnp.array(0.0)
        )
        return two["rho_xy"], parametric_to_poly(**para)["rho_xy"]

    expected, got = jax.jacfwd(pair)(0.0)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
