import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from squishyplanet.engine.parametric_ellipse import poly_to_parametric_helper
from squishyplanet.engine.planet_2d import planet_2d_coeffs

########################################################################################
# General helpers
########################################################################################


@jax.jit
def generate_sample_radii_thetas(key, num_points):
    key, subkey = jax.random.split(key)
    sample_radii = jnp.sqrt(
        jax.random.uniform(subkey, (num_points.shape[0],), minval=0, maxval=1)
    )
    key, subkey = jax.random.split(key)
    sample_thetas = jax.random.uniform(
        subkey, (num_points.shape[0],), minval=0, maxval=2 * jnp.pi
    )
    return sample_radii, sample_thetas


@jax.jit
def xy_on_surface(
    sample_radii,
    sample_thetas,
    rho_xx,
    rho_xy,
    rho_x0,
    rho_yy,
    rho_y0,
    rho_00,
    **kwargs,
):
    # n = n.shape[0]
    r1, r2, xc, yc, cosa, sina = poly_to_parametric_helper(
        rho_xx, rho_xy, rho_x0, rho_yy, rho_y0, rho_00
    )

    x = r1 * sample_radii * jnp.cos(sample_thetas)
    y = r2 * sample_radii * jnp.sin(sample_thetas)

    x_rotated = cosa * x - sina * y
    y_rotated = sina * x + cosa * y

    x_final = x_rotated + xc
    y_final = y_rotated + yc

    return x_final, y_final


@jax.jit
def z_on_surface(
    x, y, p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, **kwargs
):
    z = (
        -0.5
        * (
            p_z0
            + p_xz * x
            + p_yz * y
            - jnp.sqrt(
                (p_z0 + p_xz * x + p_yz * y) ** 2
                - 4
                * p_zz
                * (
                    -1
                    + p_00
                    + p_x0 * x
                    + p_xx * x**2
                    + p_y0 * y
                    + p_xy * x * y
                    + p_yy * y**2
                )
            )
        )
        / p_zz
    )

    return z


@jax.jit
def sample_surface(
    sample_radii,
    sample_thetas,
    rho_xx,
    rho_xy,
    rho_x0,
    rho_yy,
    rho_y0,
    rho_00,
    p_xx,
    p_xy,
    p_xz,
    p_x0,
    p_yy,
    p_yz,
    p_y0,
    p_zz,
    p_z0,
    p_00,
    **kwargs,
):
    x, y = xy_on_surface(
        sample_radii,
        sample_thetas,
        rho_xx,
        rho_xy,
        rho_x0,
        rho_yy,
        rho_y0,
        rho_00,
    )
    z = z_on_surface(
        x,
        y,
        p_xx,
        p_xy,
        p_xz,
        p_x0,
        p_yy,
        p_yz,
        p_y0,
        p_zz,
        p_z0,
        p_00,
    )
    return x, y, z


@jax.jit
def planet_surface_normal(
    x,
    y,
    z,
    p_xx,
    p_xy,
    p_xz,
    p_x0,
    p_yy,
    p_yz,
    p_y0,
    p_zz,
    p_z0,
    p_00,
):
    grad_planet = -jnp.array(
        [
            p_x0 + 2 * p_xx * x + p_xy * y + p_xz * z,
            p_y0 + p_xy * x + 2 * p_yy * y + p_yz * z,
            p_z0 + p_xz * x + p_yz * y + 2 * p_zz * z,
        ]
    )
    planet_norm = jnp.linalg.norm(grad_planet, axis=0)
    return grad_planet / planet_norm


@jax.jit
def surface_star_cos_angle(
    planet_surface_normal,
    x_c,
    y_c,
    z_c,
    **kwargs,
):

    # this is an approximation that the star is a) a point source and b) that
    # all light coming from the star is parallel. Neither of these are true. The former
    # can be handled the same way starry does it, but distributing point sources across
    # the surface of the star. I haven't seen anyone address the latter, although it
    # might be relevant for ultra-hot planets
    star = jnp.array([x_c, y_c, z_c])
    star_norm = jnp.linalg.norm(star, axis=0)

    return jnp.sum(planet_surface_normal * (star / star_norm), axis=0)


@jax.jit
def surface_observer_cos_angle(planet_surface_normal):
    # observer = jnp.array([0, 0, 1])
    return planet_surface_normal[2]


########################################################################################
# Reflection helpers
########################################################################################

def _pxx(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        p_zz * (x_c**2 + y_c**2) ** 2
        + z_c
        * (
            -(p_xz * x_c * (x_c**2 + y_c**2))
            - p_yz * y_c * (x_c**2 + y_c**2)
            + (p_xx * x_c**2 + p_xy * x_c * y_c + p_yy * y_c**2) * z_c
        )
    ) / ((x_c**2 + y_c**2) * (x_c**2 + y_c**2 + z_c**2))


def _pxy(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        -(p_yz * x_c * (x_c**2 + y_c**2))
        + p_xz * y_c * (x_c**2 + y_c**2)
        + 2 * (-p_xx + p_yy) * x_c * y_c * z_c
        + p_xy * (x_c**2 - y_c**2) * z_c
    ) / ((x_c**2 + y_c**2) * jnp.sqrt(x_c**2 + y_c**2 + z_c**2))


def _pxz(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        2
        * (
            p_xx * x_c**2
            + y_c * (p_xy * x_c + p_yy * y_c)
            - p_zz * (x_c**2 + y_c**2)
        )
        * z_c
        - p_xz * x_c * (x_c**2 + y_c**2 - z_c**2)
        - p_yz * y_c * (x_c**2 + y_c**2 - z_c**2)
    ) / (jnp.sqrt(x_c**2 + y_c**2) * (x_c**2 + y_c**2 + z_c**2))


def _px0(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (-(p_z0 * (x_c**2 + y_c**2)) + (p_x0 * x_c + p_y0 * y_c) * z_c) / (
        jnp.sqrt(x_c**2 + y_c**2) * jnp.sqrt(x_c**2 + y_c**2 + z_c**2)
    )


def _pyy(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (p_yy * x_c**2 - p_xy * x_c * y_c + p_xx * y_c**2) / (
        x_c**2 + y_c**2
    )


def _pyz(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        -2 * p_xx * x_c * y_c
        + 2 * p_yy * x_c * y_c
        + p_xy * (x_c**2 - y_c**2)
        + p_yz * x_c * z_c
        - p_xz * y_c * z_c
    ) / (
        jnp.sqrt(x_c**2 + y_c**2) * jnp.sqrt(x_c**2 + y_c**2 + z_c**2)
    )


def _py0(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (p_y0 * x_c - p_x0 * y_c) / jnp.sqrt(x_c**2 + y_c**2)


def _pzz(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (
        p_xx * x_c**2
        + p_xy * x_c * y_c
        + p_yy * y_c**2
        + p_xz * x_c * z_c
        + p_yz * y_c * z_c
        + p_zz * z_c**2
    ) / (x_c**2 + y_c**2 + z_c**2)


def _pz0(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return (p_x0 * x_c + p_y0 * y_c + p_z0 * z_c) / jnp.sqrt(
        x_c**2 + y_c**2 + z_c**2
    )


def _p00(
    p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c
):
    return p_00


@jax.jit
def planet_from_star(
    p_xx,
    p_xy,
    p_xz,
    p_x0,
    p_yy,
    p_yz,
    p_y0,
    p_zz,
    p_z0,
    p_00,
    x_c,
    y_c,
    z_c,
    **kwargs,
):
    return {
        "p_xx": _pxx(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "p_xy": _pxy(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "p_xz": _pxz(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "p_x0": _px0(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "p_yy": _pyy(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "p_yz": _pyz(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "p_y0": _py0(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "p_zz": _pzz(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "p_z0": _pz0(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
        "p_00": _p00(
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ),
    }



@jax.jit
def lambertian_reflection(surface_star_cos_angle, x, y, z):
    # return jnp.maximum(0, surface_star_angle)
    return (
        surface_star_cos_angle
        * (surface_star_cos_angle > 0)
        * ~((x**2 + y**2 < 1) & (z < 0))
    )


@jax.jit
def reflected_normalization(
    two,
    three,
    x_c,
    y_c,
    z_c,
    **kwargs,
):
    sep_squared = x_c**2 + y_c**2 + z_c**2
    # flux_density = 1 / (4 * jnp.pi * sep_squared)
    # following the starry normalization:
    flux_density = 1 / (jnp.pi * sep_squared)

    rotated_planet_3d_coeffs = planet_from_star(
        three["p_xx"],
        three["p_xy"],
        three["p_xz"],
        three["p_x0"],
        three["p_yy"],
        three["p_yz"],
        three["p_y0"],
        three["p_zz"],
        three["p_z0"],
        three["p_00"],
        x_c,
        y_c,
        z_c,
    )
    rotated_planet_2d_coeffs = planet_2d_coeffs(**rotated_planet_3d_coeffs)

    # return rotated_planet_2d_coeffs
    r1, r2, _, _, _, _ = poly_to_parametric_helper(**rotated_planet_2d_coeffs)
    area_seen_by_star = jnp.pi * r1 * r2

    return flux_density * area_seen_by_star


@jax.jit
def reflected_phase_curve(
    sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
):

    # can be used to generate just a reflected curve alone
    # if doing emission also though, some of these calculations can be reused

    if two["rho_xx"].shape != two["rho_x0"].shape:
        two["rho_xx"] = jnp.ones_like(x_c) * two["rho_xx"]
        two["rho_xy"] = jnp.ones_like(x_c) * two["rho_xy"]
        two["rho_yy"] = jnp.ones_like(x_c) * two["rho_yy"]
        three["p_xx"] = jnp.ones_like(x_c) * three["p_xx"]
        three["p_xy"] = jnp.ones_like(x_c) * three["p_xy"]
        three["p_xz"] = jnp.ones_like(x_c) * three["p_xz"]
        three["p_yy"] = jnp.ones_like(x_c) * three["p_yy"]
        three["p_yz"] = jnp.ones_like(x_c) * three["p_yz"]
        three["p_zz"] = jnp.ones_like(x_c) * three["p_zz"]

    def scan_func(carry, scan_over):
        (
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ) = scan_over
        x_c = jnp.array([x_c])
        y_c = jnp.array([y_c])
        z_c = jnp.array([z_c])

        x, y, z = sample_surface(
            sample_radii,
            sample_thetas,
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
        )
        n = planet_surface_normal(
            x,
            y,
            z,
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
        )
        surface_star_angle = surface_star_cos_angle(n, x_c, y_c, z_c)
        lamb = lambertian_reflection(surface_star_angle, x, y, z)

        return None, jnp.sum(lamb) / sample_radii.shape[0]

    flux = jax.lax.scan(
        scan_func,
        None,
        (
            two["rho_xx"],
            two["rho_xy"],
            two["rho_x0"],
            two["rho_yy"],
            two["rho_y0"],
            two["rho_00"],
            three["p_xx"],
            three["p_xy"],
            three["p_xz"],
            three["p_x0"],
            three["p_yy"],
            three["p_yz"],
            three["p_y0"],
            three["p_zz"],
            three["p_z0"],
            three["p_00"],
            x_c,
            y_c,
            z_c,
        ),
    )[1]

    norm = reflected_normalization(
        two, three, x_c, y_c, z_c
    )

    return flux * norm * state["reflected_albedo"]

########################################################################################
# Emission helpers
########################################################################################

def _x_x(a, e, f, Omega, i, omega, r, obliq, prec):
    return (
        -(jnp.sin(i) * jnp.sin(obliq) * jnp.sin(Omega))
        - jnp.cos(obliq)
        * jnp.sin(prec)
        * (
            jnp.cos(Omega) * jnp.sin(omega)
            + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
        )
        + jnp.cos(prec)
        * jnp.cos(obliq)
        * (
            jnp.cos(omega) * jnp.cos(Omega)
            - jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
        )
    )


def _x_y(a, e, f, Omega, i, omega, r, obliq, prec):
    return jnp.cos(Omega) * (
        jnp.sin(i) * jnp.sin(obliq)
        + jnp.cos(i) * jnp.cos(obliq) * jnp.sin(prec + omega)
    ) + jnp.cos(obliq) * jnp.cos(prec + omega) * jnp.sin(Omega)


def _x_z(a, e, f, Omega, i, omega, r, obliq, prec):
    return -(jnp.cos(i) * jnp.sin(obliq)) + jnp.cos(obliq) * jnp.sin(i) * jnp.sin(
        prec + omega
    )


def _x_0(a, e, f, Omega, i, omega, r, obliq, prec):
    return (a * (-1 + e**2) * jnp.cos(f - prec) * jnp.cos(obliq)) / (
        1 + e * jnp.cos(f)
    )


def _y_x(a, e, f, Omega, i, omega, r, obliq, prec):
    return -(
        jnp.cos(omega)
        * (
            jnp.cos(Omega) * jnp.sin(prec)
            + jnp.cos(i) * jnp.cos(prec) * jnp.sin(Omega)
        )
    ) + jnp.sin(omega) * (
        -(jnp.cos(prec) * jnp.cos(Omega))
        + jnp.cos(i) * jnp.sin(prec) * jnp.sin(Omega)
    )


def _y_y(a, e, f, Omega, i, omega, r, obliq, prec):
    return jnp.cos(i) * jnp.cos(prec + omega) * jnp.cos(Omega) - jnp.sin(
        prec + omega
    ) * jnp.sin(Omega)


def _y_z(a, e, f, Omega, i, omega, r, obliq, prec):
    return jnp.cos(prec + omega) * jnp.sin(i)


def _y_0(a, e, f, Omega, i, omega, r, obliq, prec):
    return (a * (-1 + e**2) * jnp.sin(f - prec)) / (1 + e * jnp.cos(f))


def _z_x(a, e, f, Omega, i, omega, r, obliq, prec):
    return (
        -(jnp.cos(Omega) * jnp.sin(prec) * jnp.sin(obliq) * jnp.sin(omega))
        + (
            jnp.cos(obliq) * jnp.sin(i)
            - jnp.cos(i) * jnp.cos(omega) * jnp.sin(prec) * jnp.sin(obliq)
        )
        * jnp.sin(Omega)
        + jnp.cos(prec)
        * jnp.sin(obliq)
        * (
            jnp.cos(omega) * jnp.cos(Omega)
            - jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
        )
    )


def _z_y(a, e, f, Omega, i, omega, r, obliq, prec):
    return -(jnp.cos(obliq) * jnp.cos(Omega) * jnp.sin(i)) + jnp.sin(obliq) * (
        jnp.cos(i) * jnp.cos(Omega) * jnp.sin(prec + omega)
        + jnp.cos(prec + omega) * jnp.sin(Omega)
    )


def _z_z(a, e, f, Omega, i, omega, r, obliq, prec):
    return jnp.cos(i) * jnp.cos(obliq) + jnp.sin(i) * jnp.sin(obliq) * jnp.sin(
        prec + omega
    )


def _z_0(a, e, f, Omega, i, omega, r, obliq, prec):
    return (a * (-1 + e**2) * jnp.cos(f - prec) * jnp.sin(obliq)) / (
        1 + e * jnp.cos(f)
    )


@jax.jit
def pre_squish_transform(a, e, f, Omega, i, omega, r, obliq, prec, **kwargs):
    mat = jnp.ones((f.shape[0], 3, 4))
    mat = mat.at[:, 0, 0].set(_x_x(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 0, 1].set(_x_y(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 0, 2].set(_x_z(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 0, 3].set(_x_0(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 1, 0].set(_y_x(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 1, 1].set(_y_y(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 1, 2].set(_y_z(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 1, 3].set(_y_0(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 2, 0].set(_z_x(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 2, 1].set(_z_y(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 2, 2].set(_z_z(a, e, f, Omega, i, omega, r, obliq, prec))
    mat = mat.at[:, 2, 3].set(_z_0(a, e, f, Omega, i, omega, r, obliq, prec))
    return mat

@jax.jit
def _emission_profle(x, y, z, r, f1, f2, hotspot_latitude, hotspot_longitude, hotspot_concentration):
    return (
        jnp.exp(
            hotspot_concentration
            + (
                hotspot_concentration
                * (
                    -((-1 + f2) * z * jnp.cos(hotspot_latitude))
                    + (-1 + f1)
                    * jnp.sin(hotspot_latitude)
                    * ((-1 + f2) * x * jnp.cos(hotspot_longitude) - y * jnp.sin(hotspot_longitude))
                )
            )
            / (
                (-1 + f1)
                * (-1 + f2)
                * r
                * jnp.sqrt(
                    (x**2 + y**2 / (-1 + f2) ** 2 + z**2 / (-1 + f1) ** 2)
                    / r**2
                )
            )
        )
        * hotspot_concentration
    ) / (-2 * jnp.pi + 2 * jnp.exp(2 * hotspot_concentration) * jnp.pi)


@jax.jit
def emission_profile(
    x,
    y,
    z,
    transform,
    r,
    f1,
    f2,
    hotspot_latitude,
    hotspot_longitude,
    hotspot_concentration,
    **kwargs
):
    # always one time slice at a time

    # do this check before you transform into the planet frame
    mask = ~((x**2 + y**2 < 1) & (z < 0))
    x, y, z = jnp.matmul(transform, jnp.array([x, y, z, jnp.ones_like(x)]))
    return _emission_profle(x, y, z, r, f1, f2, hotspot_latitude, hotspot_longitude, hotspot_concentration) * mask



@jax.jit
def emission_phase_curve(
    sample_radii,
    sample_thetas,
    two,
    three,
    state,
    **kwargs,
):
    # can be used to generate just an emitted phase curve alone
    # if doing reflection also though, some of these calculations can be reused

    if two["rho_xx"].shape != two["rho_x0"].shape:
        two["rho_xx"] = jnp.ones_like(two["rho_x0"]) * two["rho_xx"]
        two["rho_xy"] = jnp.ones_like(two["rho_x0"]) * two["rho_xy"]
        two["rho_yy"] = jnp.ones_like(two["rho_x0"]) * two["rho_yy"]
        three["p_xx"] = jnp.ones_like(two["rho_x0"]) * three["p_xx"]
        three["p_xy"] = jnp.ones_like(two["rho_x0"]) * three["p_xy"]
        three["p_xz"] = jnp.ones_like(two["rho_x0"]) * three["p_xz"]
        three["p_yy"] = jnp.ones_like(two["rho_x0"]) * three["p_yy"]
        three["p_yz"] = jnp.ones_like(two["rho_x0"]) * three["p_yz"]
        three["p_zz"] = jnp.ones_like(two["rho_x0"]) * three["p_zz"]

    def scan_func(carry, scan_over):
        (
            transform_matrix,
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
        ) = scan_over

        x, y, z = sample_surface(
            sample_radii,
            sample_thetas,
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
        )
        emission_samples = emission_profile(
            x,
            y,
            z,
            transform_matrix,
            state["r"],
            state["f1"],
            state["f2"],
            state["hotspot_latitude"],
            state["hotspot_longitude"],
            state["hotspot_concentration"],
        )

        return None, jnp.sum(emission_samples) / emission_samples.shape[0]

    transform_matricies = pre_squish_transform(**state)

    flux = jax.lax.scan(
        scan_func,
        None,
        (
            transform_matricies,
            two["rho_xx"],
            two["rho_xy"],
            two["rho_x0"],
            two["rho_yy"],
            two["rho_y0"],
            two["rho_00"],
            three["p_xx"],
            three["p_xy"],
            three["p_xz"],
            three["p_x0"],
            three["p_yy"],
            three["p_yz"],
            three["p_y0"],
            three["p_zz"],
            three["p_z0"],
            three["p_00"],
        ),
    )[1]

    return flux * state["emitted_scale"]


########################################################################################
# Combined curves
########################################################################################

@jax.jit
def phase_curve(
    sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
):

    if two["rho_xx"].shape != two["rho_x0"].shape:
        two["rho_xx"] = jnp.ones_like(x_c) * two["rho_xx"]
        two["rho_xy"] = jnp.ones_like(x_c) * two["rho_xy"]
        two["rho_yy"] = jnp.ones_like(x_c) * two["rho_yy"]
        three["p_xx"] = jnp.ones_like(x_c) * three["p_xx"]
        three["p_xy"] = jnp.ones_like(x_c) * three["p_xy"]
        three["p_xz"] = jnp.ones_like(x_c) * three["p_xz"]
        three["p_yy"] = jnp.ones_like(x_c) * three["p_yy"]
        three["p_yz"] = jnp.ones_like(x_c) * three["p_yz"]
        three["p_zz"] = jnp.ones_like(x_c) * three["p_zz"]

    def scan_func(carry, scan_over):
        (
            transform_matrix,
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
            x_c,
            y_c,
            z_c,
        ) = scan_over
        x_c = jnp.array([x_c])
        y_c = jnp.array([y_c])
        z_c = jnp.array([z_c])

        x, y, z = sample_surface(
            sample_radii,
            sample_thetas,
            rho_xx,
            rho_xy,
            rho_x0,
            rho_yy,
            rho_y0,
            rho_00,
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
        )

        # reflection stuff
        n = planet_surface_normal(
            x,
            y,
            z,
            p_xx,
            p_xy,
            p_xz,
            p_x0,
            p_yy,
            p_yz,
            p_y0,
            p_zz,
            p_z0,
            p_00,
        )
        surface_star_angle = surface_star_cos_angle(n, x_c, y_c, z_c)
        lamb = lambertian_reflection(surface_star_angle, x, y, z)

        # emission stuff
        emission_samples = emission_profile(
            x,
            y,
            z,
            transform_matrix,
            state["r"],
            state["f1"],
            state["f2"],
            state["hotspot_latitude"],
            state["hotspot_longitude"],
            state["hotspot_concentration"],
        )

        return None, (jnp.sum(lamb) / sample_radii.shape[0], jnp.sum(emission_samples) / emission_samples.shape[0])

    transform_matricies = pre_squish_transform(**state)
    
    fluxes = jax.lax.scan(
        scan_func,
        None,
        (
            transform_matricies,
            two["rho_xx"],
            two["rho_xy"],
            two["rho_x0"],
            two["rho_yy"],
            two["rho_y0"],
            two["rho_00"],
            three["p_xx"],
            three["p_xy"],
            three["p_xz"],
            three["p_x0"],
            three["p_yy"],
            three["p_yz"],
            three["p_y0"],
            three["p_zz"],
            three["p_z0"],
            three["p_00"],
            x_c,
            y_c,
            z_c,
        ),
    )[1]

    reflected_norm = reflected_normalization(
        two, three, x_c, y_c, z_c
    )

    return fluxes[0] * reflected_norm * state["reflected_albedo"], fluxes[1] * state["emitted_scale"] # the reflected and emitted contributions