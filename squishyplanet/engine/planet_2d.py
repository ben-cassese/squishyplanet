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
    """
    Compute the coefficients that describe the planet as an implicit 2D surface from the
    observer's perspective.

    This function transforms the coefficients that describe the planet's 3D shape into
    a new set of coefficients that describe its projected outline as seen from
    z=infinity. The input p coefficients satisfy the equation:

    .. math::
                p_{xx} x^2 + p_{xy} xy + p_{xz} xz + p_{x0} x + p_{yy} y^2 + p_{yz} yz + p_{y0} y + p_{zz} z^2 + p_{z0} z + p_{00} = 1

    Args:
        p_xx (Array): Coefficient representing xx interaction in the 3D model.
        p_xy (Array): Coefficient representing xy interaction in the 3D model.
        p_xz (Array): Coefficient representing xz interaction in the 3D model.
        p_x0 (Array): Coefficient representing x0 interaction in the 3D model.
        p_yy (Array): Coefficient representing yy interaction in the 3D model.
        p_yz (Array): Coefficient representing yz interaction in the 3D model.
        p_y0 (Array): Coefficient representing y0 interaction in the 3D model.
        p_zz (Array): Coefficient representing zz interaction in the 3D model.
        p_z0 (Array): Coefficient representing z0 interaction in the 3D model.
        p_00 (Array): Coefficient representing 00 interaction in the 3D model.
        **kwargs: Unused additional keyword arguments. These are included to allow for
                flexibility in providing additional data that may be ignored during the
                computation but included for interface consistency.

    Returns:
        dict:
            A dictionary with keys representing different transformed coefficient
            names ('rho_xx', 'rho_xy', 'rho_x0', 'rho_yy', 'rho_y0', 'rho_00') and
            their corresponding values. These coefficients describe the outline of the
            planet as an implicit curve that satisfies the equation:

            .. math::
                \\rho_{xx} x^2 + \\rho_{xy} xy + \\rho_{x0} x + \\rho_{yy} y^2 + \\rho_{y0} y + \\rho_{00} = 1
    """
    return {
        "rho_xx": _rho_xx(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00),
        "rho_xy": _rho_xy(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00),
        "rho_x0": _rho_x0(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00),
        "rho_yy": _rho_yy(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00),
        "rho_y0": _rho_y0(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00),
        "rho_00": _rho_00(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00),
    }
