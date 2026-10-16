"""A projected ellipse tilted by an odd multiple of 45 degrees has rho_xx == rho_yy but is
not a circle; its tilt used to be zeroed, giving ~1e-2 flux errors."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from squishyplanet import OblateSystem
from squishyplanet.engine.parametric_ellipse import ellipse_angle

TIMES = jnp.linspace(-0.12, 0.12, 401)


def _system(theta: float, f: float) -> OblateSystem:
    return OblateSystem(
        times=TIMES,
        t_peri=-0.75,
        period=3.0,
        a=8.0,
        e=0.0,
        i=float(np.arccos(0.3 / 8.0)),
        omega=0.0,
        tidally_locked=False,
        ld_u_coeffs=jnp.array([0.3, 0.2]),
        parameterize_with_projected_ellipse=True,
        projected_effective_r=0.1,
        projected_f=f,
        projected_theta=theta,
    )


@pytest.mark.parametrize("f", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("k", [1, 3, 5, 7, -1])
def test_lightcurve_is_continuous_across_odd_multiples_of_45_degrees(
    k: int, f: float
) -> None:
    theta = k * jnp.pi / 4
    at = np.asarray(_system(theta, f).lightcurve({}))
    near = np.asarray(_system(theta + 1e-7, f).lightcurve({}))
    # moving the angle by 1e-7 rad changes the flux by << 0.01 ppm
    assert np.abs(at - near).max() < 1e-8


def test_ellipse_angle_tilted_with_equal_diagonal_terms() -> None:
    # 2x^2 + xy + 2y^2 = 1 has its major axis along (1, -1), i.e. at 135 degrees
    assert jnp.isclose(ellipse_angle(2.0, 1.0, 2.0), 3 * jnp.pi / 4)
    assert jnp.isclose(ellipse_angle(2.0, -1.0, 2.0), jnp.pi / 4)


def test_ellipse_angle_circle_is_zero_with_finite_gradient() -> None:
    assert ellipse_angle(3.0, 0.0, 3.0) == 0.0
    grad = jax.grad(ellipse_angle, argnums=(0, 1, 2))(3.0, 0.0, 3.0)
    assert all(np.isfinite(g) for g in grad)


def test_gradient_wrt_theta_is_finite_at_45_degrees() -> None:
    system = _system(jnp.pi / 4, 0.2)

    def total_flux(theta: jax.Array) -> jax.Array:
        return jnp.sum(system.lightcurve({"projected_theta": theta}))

    assert np.isfinite(jax.grad(total_flux)(jnp.pi / 4))
