import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def _rho_xx(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_xx - p_xz**2 / (4.0 * p_zz)


def _rho_xy(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_xy - (p_xz * p_yz) / (2.0 * p_zz)


def _rho_x0(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_x0 - (p_xz * p_z0) / (2.0 * p_zz)


def _rho_yy(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_yy - p_yz**2 / (4.0 * p_zz)


def _rho_y0(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_y0 - (p_yz * p_z0) / (2.0 * p_zz)


def _rho_00(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00):
    return p_00 - p_z0**2 / (4.0 * p_zz)


@jax.jit
def planet_2d_coeffs(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, **kwargs
):
    return {
        "rho_xx": _rho_xx(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_xy": _rho_xy(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_x0": _rho_x0(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_yy": _rho_yy(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_y0": _rho_y0(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
        "rho_00": _rho_00(
            p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00
        ),
    }
