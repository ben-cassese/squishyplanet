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
    """
    Create a random set of radii and thetas for sampling the planet's surface.

    These are uniformly distributed through a unit circle and will be scaled and rotated
    to match the planet's shape and orientation at each timestep. However, they will
    be re-used at every time step, which could introduce a bias but makes things much
    faster. Be sure you use sufficient samples to keep the bias small, then try multiple
    random keys to quantify it.

    Args:
        key (Array): A jax.random.PRNGKey for generating random numbers.
        num_points (int): The number of points to generate.

    Returns:
        Tuple:
            A tuple of two arrays, the first containing the radii and the second
            containing the thetas.
    """
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
def _xy_on_surface(
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
def _z_on_surface(
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
    """
    Convert randomly sampled :math:`(x, y)` points on the projected planet to
    :math:`(x, y, z)` points on the planet's surface.

    The :math:`rho` coefficients are calculated with :func:`planet_2d.planet_2d_coeffs`,
    the :math:`p` coefficients are calculated with :func:`planet_3d.planet_3d_coeffs`,
    and the sample points are generated with :func:`generate_sample_radii_thetas`.

    Args:
        sample_radii (Array): The radii of the sampled points.
        sample_thetas (Array): The angles of the sampled points.
        rho_xx (Array): xx coefficient in the 2D implicit representation.
        rho_xy (Array): xy coefficient in the 2D implicit representation.
        rho_x0 (Array): x0 coefficient in the 2D implicit representation.
        rho_yy (Array): yy coefficient in the 2D implicit representation.
        rho_y0 (Array): y0 coefficient in the 2D implicit representation.
        rho_00 (Array): 00 coefficient in the 2D implicit representation.
        p_xx (Array): xx coefficient in the 3D implicit representation.
        p_xy (Array): xy coefficient in the 3D implicit representation.
        p_xz (Array): xz coefficient in the 3D implicit representation.
        p_x0 (Array): x0 coefficient in the 3D implicit representation.
        p_yy (Array): yy coefficient in the 3D implicit representation.
        p_yz (Array): yz coefficient in the 3D implicit representation.
        p_y0 (Array): y0 coefficient in the 3D implicit representation.
        p_zz (Array): zz coefficient in the 3D implicit representation.
        p_z0 (Array): z0 coefficient in the 3D implicit representation.
        p_00 (Array): 00 coefficient in the 3D implicit representation.
        **kwargs:
            Additional unused keyword arguments, included so that we can pass in
            a larger state dictionary that includes all of the required parameters along
            with other unnecessary ones.

    Returns:
        Tuple:
            A tuple of three arrays, the first containing the x values, the second
            containing the y values, and the third containing the z values.

    """
    x, y = _xy_on_surface(
        sample_radii,
        sample_thetas,
        rho_xx,
        rho_xy,
        rho_x0,
        rho_yy,
        rho_y0,
        rho_00,
    )
    z = _z_on_surface(
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
    """
    Compute the unit normal vector to the planet's surface at a given point.

    The input :math:`(x, y, z)` points are assumed to lie on the planet's surface.

    Args:
        x (Array): The x values of the points.
        y (Array): The y values of the points.
        z (Array): The z values of the points.
        p_xx (Array): xx coefficient in the 3D implicit representation.
        p_xy (Array): xy coefficient in the 3D implicit representation.
        p_xz (Array): xz coefficient in the 3D implicit representation.
        p_x0 (Array): x0 coefficient in the 3D implicit representation.
        p_yy (Array): yy coefficient in the 3D implicit representation.
        p_yz (Array): yz coefficient in the 3D implicit representation.
        p_y0 (Array): y0 coefficient in the 3D implicit representation.
        p_zz (Array): zz coefficient in the 3D implicit representation.
        p_z0 (Array): z0 coefficient in the 3D implicit representation.
        p_00 (Array): 00 coefficient in the 3D implicit representation.

    Returns:
        Array:
            An array of shape (3, n) containing the unit normal vectors at each point.
    """
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
    """
    A helper function to compute the cosine of the angle between the planet's surface
    normal vector and the vector linking the center of the planet to the star.

    This is an approximation that the star is a) a point source and b) that
    all light coming from the star is parallel. Neither of these are strictly true. The
    former could be handled the same way starry does it, by distributing point sources
    across the surface of the star and averaging. I don't know of any attempts to
    address the latter, though in principle it wouldn't be hard to do here since we're
    already doing so much numerically.

    Args:
        planet_surface_normal (Array): The unit normal vectors to the planet's surface.
        x_c (Array): The x coordinate of the center of the planet.
        y_c (Array): The y coordinate of the center of the planet.
        z_c (Array): The z coordinate of the center of the planet.
        **kwargs:
            Additional unused keyword arguments, included so that we can pass in
            a larger state dictionary that includes all of the required parameters along
            with other unnecessary ones.

    Returns:
        Array:
            The cosine of the angle between the planet's surface normal and the vector
            pointing from the planet's center to the star.


    """

    star = jnp.array([x_c, y_c, z_c])
    star_norm = jnp.linalg.norm(star, axis=0)

    return jnp.sum(planet_surface_normal * (star / star_norm), axis=0)


@jax.jit
def _surface_observer_cos_angle(planet_surface_normal):
    # observer = jnp.array([0, 0, 1])
    return planet_surface_normal[2]


########################################################################################
# Reflection helpers
########################################################################################


def _pxx(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
    return (
        p_zz * (x_c**2 + y_c**2) ** 2
        + z_c
        * (
            -(p_xz * x_c * (x_c**2 + y_c**2))
            - p_yz * y_c * (x_c**2 + y_c**2)
            + (p_xx * x_c**2 + p_xy * x_c * y_c + p_yy * y_c**2) * z_c
        )
    ) / ((x_c**2 + y_c**2) * (x_c**2 + y_c**2 + z_c**2))


def _pxy(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
    return (
        -(p_yz * x_c * (x_c**2 + y_c**2))
        + p_xz * y_c * (x_c**2 + y_c**2)
        + 2 * (-p_xx + p_yy) * x_c * y_c * z_c
        + p_xy * (x_c**2 - y_c**2) * z_c
    ) / ((x_c**2 + y_c**2) * jnp.sqrt(x_c**2 + y_c**2 + z_c**2))


def _pxz(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
    return (
        2
        * (p_xx * x_c**2 + y_c * (p_xy * x_c + p_yy * y_c) - p_zz * (x_c**2 + y_c**2))
        * z_c
        - p_xz * x_c * (x_c**2 + y_c**2 - z_c**2)
        - p_yz * y_c * (x_c**2 + y_c**2 - z_c**2)
    ) / (jnp.sqrt(x_c**2 + y_c**2) * (x_c**2 + y_c**2 + z_c**2))


def _px0(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
    return (-(p_z0 * (x_c**2 + y_c**2)) + (p_x0 * x_c + p_y0 * y_c) * z_c) / (
        jnp.sqrt(x_c**2 + y_c**2) * jnp.sqrt(x_c**2 + y_c**2 + z_c**2)
    )


def _pyy(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
    return (p_yy * x_c**2 - p_xy * x_c * y_c + p_xx * y_c**2) / (x_c**2 + y_c**2)


def _pyz(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
    return (
        -2 * p_xx * x_c * y_c
        + 2 * p_yy * x_c * y_c
        + p_xy * (x_c**2 - y_c**2)
        + p_yz * x_c * z_c
        - p_xz * y_c * z_c
    ) / (jnp.sqrt(x_c**2 + y_c**2) * jnp.sqrt(x_c**2 + y_c**2 + z_c**2))


def _py0(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
    return (p_y0 * x_c - p_x0 * y_c) / jnp.sqrt(x_c**2 + y_c**2)


def _pzz(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
    return (
        p_xx * x_c**2
        + p_xy * x_c * y_c
        + p_yy * y_c**2
        + p_xz * x_c * z_c
        + p_yz * y_c * z_c
        + p_zz * z_c**2
    ) / (x_c**2 + y_c**2 + z_c**2)


def _pz0(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
    return (p_x0 * x_c + p_y0 * y_c + p_z0 * z_c) / jnp.sqrt(x_c**2 + y_c**2 + z_c**2)


def _p00(p_xx, p_xy, p_xz, p_x0, p_yy, p_yz, p_y0, p_zz, p_z0, p_00, x_c, y_c, z_c):
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
    """
    Compute the coefficients of the planet's 3D shape from the star's perspective,
    as if it were aligned the the :math:`z` axis.

    When computing the reflected flux from the planet, we need to know how much flux
    initial reaches it from the star. To do that, we need to know the planet's projected
    area as seen from the star, which importantly, could be different than the projected
    area as seen from the observer. To compute this area, we first use this function to
    get a 3D representation of the planet as seen from the star, then will use those
    coefficients to compute an implicit 2D representation, then will use those to get
    the area.

    The x_c, y_c, and z_c inputs are all technically encoded in the p inputs as well,
    but it was easier just to carry them around explicitly.

    Args:
        p_xx (Array): xx coefficient in the 3D implicit representation.
        p_xy (Array): xy coefficient in the 3D implicit representation.
        p_xz (Array): xz coefficient in the 3D implicit representation.
        p_x0 (Array): x0 coefficient in the 3D implicit representation.
        p_yy (Array): yy coefficient in the 3D implicit representation.
        p_yz (Array): yz coefficient in the 3D implicit representation.
        p_y0 (Array): y0 coefficient in the 3D implicit representation.
        p_zz (Array): zz coefficient in the 3D implicit representation.
        p_z0 (Array): z0 coefficient in the 3D implicit representation.
        p_00 (Array): 00 coefficient in the 3D implicit representation.
        x_c (Array): The x coordinate of the center of the planet.
        y_c (Array): The y coordinate of the center of the planet.
        z_c (Array): The z coordinate of the center of the planet.

    Returns:
        dict:
            A dictionary containing the coefficients of the planet's shape as seen from
            the star. Will look identical to the output of :func:`planet_3d.planet_3d_coeffs`.

    """
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
    """
    Compute the reflected intensity at a specific point on the planet's surface assuming
    a simple Lambertian reflection model.

    This is a simple model that assumes the planet reflects light according to Lambert's
    cosine law, which states that the intensity of reflected light is proportional to
    the cosine of the angle between the surface normal and the illumination direction.
    That arrangement means it does *not* depend on the observer's viewing angle, only
    the illumination angle. This helper function also assumes a uniform albedo of 1
    across the planet's surface (the final reflected flux will be scaled by the provided
    albedo, though is still always assumed to be uniform).

    This function will also mask out any points on the planet's surface that are on the
    wrong side of the terminator.

    Args:
        surface_star_cos_angle (Array): The cosine of the angle between the planet's
            surface normal and the vector pointing from the planet's center to the star.
        x (Array): The x values of the points on the planet's surface.
        y (Array): The y values of the points on the planet's surface.
        z (Array): The z values of the points on the planet's surface.

    Returns:
        Array: The intensity of the reflected light at each point.

    """
    # return jnp.maximum(0, surface_star_angle)
    return surface_star_cos_angle * (surface_star_cos_angle > 0)


@jax.jit
def _henyey_greenstein(g, theta):
    return 0.5 * (1 - g1**2) / (1 + g1**2 - 2 * g * jnp.cos(theta)) ** (1.5)


@jax.jit
def _two_term_henyey_greenstein(gf, gb, scatter_f, theta):
    return scatter_f * _henyey_greenstein(gf, theta) + (
        1 - scatter_f
    ) * _henyey_greenstein(gb, theta)


@jax.jit
def _rayleigh_scattering(theta):
    return (3 / 4) * (1 + jnp.cos(theta) ** 2)


@jax.jit
def reflected_normalization(
    two,
    three,
    x_c,
    y_c,
    z_c,
    xo=0.0,
    yo=0.0,
    zo=0.0,
    **kwargs,
):
    """
    Compute the time-dependent normalization factor for the reflected light.

    The reflected light computations are almost entirely carried out assuming the star
    is a point source 1 R_star from the center of the planet emitting plane-parallel
    rays. To convert these to actual reflected flux, we need to a) correct for the
    distance between the planet and the star and b) account for how much area the planet
    actually subtends as seen from the star. a) is easy and common across all
    implementations, it's just the inverse square law. b) is more complicated for oblate
    planets than spherical planets however, since even on circular orbits, the subtended
    area (and consequently area that is recieves flux and and is able to reflect it) can
    change as a function of orbital phase. Note however that it will not vary with phase
    if the planet is tidally locked and always shows the same face to the star.

    Args:
        two (dict):
            A dictionary containing the rho coefficients of the planet's
            implicit 2D representation, as seen from the observer and calculated with
            :func:`planet_2d.planet_2d_coeffs`.
        three (dict):
            A dictionary containing the p coefficients of the planet's 3D shape, as seen
            from the observer and calculated with :func:`planet_3d.planet_3d_coeffs`.
        x_c (Array): The x coordinate of the center of the planet.
        y_c (Array): The y coordinate of the center of the planet.
        z_c (Array): The z coordinate of the center of the planet.
        xo (float or Array):
            An offset to add to the x coordinate of the center of the planet, used when
            correcting for extended source illuminations. Default is 0.0.
        yo (float or Array):
            An offset to add to the y coordinate of the center of the planet, used when
            correcting for extended source illuminations. Default is 0.0.
        zo (float or Array):
            An offset to add to the z coordinate of the center of the planet, used when
            correcting for extended source illuminations. Default is 0.0.
        **kwargs:
            Additional unused keyword arguments, included so that we can pass in
            a larger state dictionary that includes all of the required parameters along
            with other unnecessary ones.

    Returns:
        Array: The normalization factor for the reflected light.

    """
    x_c = x_c - xo
    y_c = y_c - yo
    z_c = z_c - zo

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

    # # return rotated_planet_2d_coeffs
    r1, r2, _, _, _, _ = poly_to_parametric_helper(**rotated_planet_2d_coeffs)
    area_seen_by_star = jnp.pi * r1 * r2

    return flux_density * area_seen_by_star


@jax.jit
def reflected_phase_curve(
    sample_radii,
    sample_thetas,
    two,
    three,
    state,
    x_c,
    y_c,
    z_c,
    xo=jnp.array([0.0]),
    yo=jnp.array([0.0]),
    zo=jnp.array([0.0]),
):
    """
    Compute the timeseries of light reflected from  the planet.

    This function computes the reflected light from the planet at each time step. It
    assume the planet is a) a Lambertian reflector, b) that the star is a point source
    sending out parallel rays, c) that the :math:`a/R_s >> R_p` (i.e., the distance
    between each point on the surface to the star is essentially constant), and d) that
    the planet has a spatially uniform albedo of unity (the entire curve can be scaled
    by an actual albedo later, but the assumption of uniformity is baked-in). However,
    it does take into account the planet's oblateness and orientation.

    Args:
        sample_radii (Array):
            Randomly sampled radii from a unit sphere, used to generate points on the
            visible disk of the planet at each timestep. Create with
            :func:`generate_sample_radii_thetas`.
        sample_thetas (Array):
            Randomly sampled thetas from a unit sphere, used to generate points on the
            visible disk of the planet at each timestep. Create with
            :func:`generate_sample_radii_thetas`.
        two (dict):
            A dictionary containing the rho coefficients of the planet's implicit 2D
            representation, as seen from the observer and calculated with
            :func:`planet_2d.planet_2d_coeffs`.
        three (dict):
            A dictionary containing the p coefficients of the planet's 3D shape, as seen
            from the observer and calculated with :func:`planet_3d.planet_3d_coeffs`.
        state (dict):
            A dictionary containing all of the parameters needed to compute the phase
            curve. This includes the planet's orbital parameters, the observer's
            parameters, and the hotspot parameters.
        x_c (Array): The x coordinate of the center of the planet.
        y_c (Array): The y coordinate of the center of the planet.
        z_c (Array): The z coordinate of the center of the planet.
        xo (Array):
            An offset to add to the x coordinate of the center of the planet, used when
            correcting for extended source illuminations. Default is 0.0.
        yo (Array):
            An offset to add to the y coordinate of the center of the planet, used when
            correcting for extended source illuminations. Default is 0.0.
        zo (Array):
            An offset to add to the z coordinate of the center of the planet, used when
            correcting for extended source illuminations. Default is 0.0.


    Returns:
        Array:
            The timeseries of reflected light from the planet. Each element of the array
            corresponds to the time of the corresponding element in state["times"].


    """

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

    if x_c.shape != xo.shape:
        xo = jnp.ones_like(x_c) * xo
        yo = jnp.ones_like(x_c) * yo
        zo = jnp.ones_like(x_c) * zo

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
            xo,
            yo,
            zo,
        ) = scan_over
        x_c = jnp.array([x_c]) - xo
        y_c = jnp.array([y_c]) - yo
        z_c = jnp.array([z_c]) - zo

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

        mask = ~(((x + xo) ** 2 + (y + yo) ** 2 < 1) & ((z + zo) < 0))
        lamb = lamb * mask

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
            xo,
            yo,
            zo,
        ),
    )[1]

    norm = reflected_normalization(two, three, x_c, y_c, z_c, xo, yo, zo)

    return flux * norm * state["albedo"]


# still having trouble with this, leaving it for a specific enhancement after 0.1.0
@jax.jit
def extended_illumination_reflected_phase_curve(
    sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c, offsets
):
    """
    WIP, not yet implemented. Hiding behind a NotImplementedError when setting
    `extended_illumination_npts` to anything greater than 1 when initializing an
    :class:`OblateSystem` object.
    """
    pass
    # def scan_func(carry, scan_over):
    #     two, three = scan_over
    #     return None, reflected_phase_curve(
    #         sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
    #     )

    # reflected = jax.lax.scan(scan_func,None,(two, three))[1]
    # return jnp.mean(reflected, axis=0)

    # xo, yo, zo = offsets[..., 0], offsets[..., 1], offsets[..., 2]
    # reflected = jax.vmap(
    #     reflected_phase_curve,
    #     in_axes=(None, None, 0, 0, None, None, None, None, 0, 0, 0),
    # )(sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c, xo, yo, zo)
    # # return jnp.mean(reflected, axis=0)
    # return reflected


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
    return (a * (-1 + e**2) * jnp.cos(f - prec) * jnp.cos(obliq)) / (1 + e * jnp.cos(f))


def _y_x(a, e, f, Omega, i, omega, r, obliq, prec):
    return -(
        jnp.cos(omega)
        * (jnp.cos(Omega) * jnp.sin(prec) + jnp.cos(i) * jnp.cos(prec) * jnp.sin(Omega))
    ) + jnp.sin(omega) * (
        -(jnp.cos(prec) * jnp.cos(Omega)) + jnp.cos(i) * jnp.sin(prec) * jnp.sin(Omega)
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
    return (a * (-1 + e**2) * jnp.cos(f - prec) * jnp.sin(obliq)) / (1 + e * jnp.cos(f))


@jax.jit
def pre_squish_transform(a, e, f, Omega, i, omega, r, obliq, prec, **kwargs):
    """
    Compute the rotation matrix to go from the sky frame to the planet's.

    This is the underlying transformation behind everything in
    :func:`planet_3d.planet_3d_coeffs`, except that module never actually uses it
    in this form since it applies it then goes ahead and gathers terms.

    Args:
        a (Array): The semi-major axis of the planet.
        e (Array): The eccentricity of the planet.
        f (Array): The true anomaly of the planet.
        Omega (Array): The longitude of the ascending node of the planet.
        i (Array): The inclination of the planet.
        omega (Array): The argument of periapsis of the planet.
        r (Array): The equatorial radius of the planet.
        obliq (Array): The obliquity of the planet.
        prec (Array): The precession of the planet.
        **kwargs:
            Additional unused keyword arguments, included so that we can pass in
            a larger state dictionary that includes all of the required parameters along
            with other unnecessary ones.

    Returns:
        Array:
            A matrix that can be used to rotate vectors from the sky frame to the
            planet's frame.

    """
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
def _uncorrected_emission_profile(
    x, y, z, r, f1, f2, hotspot_latitude, hotspot_longitude, hotspot_concentration
):
    # this will produce and UNNORMALIZED emission sample- we aren't correcting for the
    # sphere-to-ellipsoid mapping yet

    # When randomly sampling over a sphere, this will get you an average value of 1

    # the (4(jnp.pi)) is also 1 / float(gamma(3/2) / (2*jnp.pi**(3/2))), is called out
    # here, used to convert from flux/area to flux
    # https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution

    # first, inflate the x,y,z samples (which live on the planet's surface)
    # to the unit sphere
    x = x / r
    y = y / (r * (1 - f2))
    z = z / (r * (1 - f1))

    # then, evalutate the pdf of the von Mises-Fisher distribution:
    return (
        (
            jnp.exp(
                (
                    hotspot_concentration
                    * (
                        z * jnp.cos(hotspot_latitude)
                        + x * jnp.cos(hotspot_longitude) * jnp.sin(hotspot_latitude)
                        + y * jnp.sin(hotspot_latitude) * jnp.sin(hotspot_longitude)
                    )
                )
                / 1.0
            )
            * hotspot_concentration
        )
        / (
            2.0
            * (-jnp.exp(-hotspot_concentration) + jnp.exp(hotspot_concentration))
            * jnp.pi
        )
        * 12.566370614359174
    )


# @jax.jit
# def emission_squish_correction(x, y, z, r, f1, f2):
#     """
#     Correction factor for the squishing of the planet due to its oblateness.

#     We're using the
#     `von Mises-Fisher distribution
#     <https://https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution>`_ to
#     model a hotspot. But, that's defined on the unit sphere, and after compressing it to
#     the squished planet, the surface denisty of emission intensity will be warped. We
#     need to correct for that warping here. The input coordinates here are **IN THE
#     PLANET'S FRAME, NOT THE SKY FRAME.** After getting :math:`x,y,z` samples in the sky
#     frame, apply the rotation matrix from :func:`pre_squish_transform` to get these.

#     This is based on Algorithm 1 of Marples and Williams 2024 `doi:10.1007/s11075-023-01628-4
#     <https://doi.org/10.1007/s11075-023-01628-4>`_, which is a rejection-sampling scheme
#     for drawing points uniformly from the surface of an ellipsoid. In this case, our
#     samples are not uniformly distributed, since we sampled evenly on the the projected
#     disk, not the 3D surface. But, since we're still mapping from the planet to the unit
#     sphere when evaluating the von Mises-Fisher distribution, we need to correct for the
#     geometry-induced warping. We do that by weighting each point by the inverse of the
#     probability that they would have been rejected when mapping the unit sphere to the
#     planet.

#     Args:
#         x (Array):
#             The x values of the points on the planet's surface IN THE PLANET'S FRAME
#         y (Array):
#             The y values of the points on the planet's surface IN THE PLANET'S FRAME
#         z (Array):
#             The z values of the points on the planet's surface IN THE PLANET'S FRAME
#         r (Array):
#             The equatorial radius of the planet.
#         f1 (Array):
#             The planet's :math:`z` flattening coefficient.
#         f2 (Array):
#             The planet's :math:`y` flattening coefficient.

#     Returns:
#         Array:
#             The correction factor for the squishing of the planet due to its oblateness.


#     """
#     a = r
#     b = jnp.sqrt(r**2 * (1 - f2) ** 2)
#     c = jnp.sqrt(r**2 * (1 - f1) ** 2)

#     # c assumed to be the smallest axis
#     g = c * jnp.sqrt(x**2 / a**4 + y**2 / b**4 + z**2 / c**4)

#     weight = 1.0 / g
#     return weight


@jax.jit
def corrected_emission_profile(
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
    **kwargs,
):
    """
    A helper function to :func:`emission_at_timestep`, broken out only to be used for
    illustrations in :func:`OblateSystem.illustrate`.
    """

    # always one time slice at a time

    # do this check before you transform into the planet frame
    mask = ~((x**2 + y**2 < 1) & (z < 0))
    x, y, z = jnp.matmul(transform, jnp.array([x, y, z, jnp.ones_like(x)]))

    #
    # correction = emission_squish_correction(x, y, z, r, f1, f2)

    # _uncorrected_emission_profile takes the samples on the planet surface and boosts them onto
    # a unit sphere. Had the samples been uniformly distributed on the planet's surface,
    # applying the correction factor to make up for the squishing would be enough. But,
    # we sampled uniformly on the projected disk, not the 3D surface: they don't occupy
    # an area of 4pi, just pi. So, we divide by another factor of 4 here.

    # To check this, create a super-concentrated hotspot on a tidally locked planet. The
    # peak emission should be nearly equal to emitted_scale when the hotspot faces the
    # observer, since the rest of the contributions are negligible and the area isn't
    # distorted by viewing geometry

    return (
        (
            _uncorrected_emission_profile(
                x,
                y,
                z,
                r,
                f1,
                f2,
                hotspot_latitude,
                hotspot_longitude,
                hotspot_concentration,
            )
            * mask
        )
        / x.shape[0]
        #  * correction
        #  / jnp.sum(correction)
        #  / 4.0
    )


@jax.jit
def emission_at_timestep(
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
):
    """
    Compute the emitted intensity at a given point on the planet's surface.

    Args:
        x (Array):
            The x values of the points on the planet's surface in the sky frame
        y (Array):
            The y values of the points on the planet's surface in the sky frame
        z (Array):
            The z values of the points on the planet's surface in the sky frame
        transform (Array):
            The rotation matrix to transform the sky frame to the planet's frame,
            calculated with :func:`pre_squish_transform`.
        r (Array):
            The equatorial radius of the planet.
        f1 (Array):
            The planet's :math:`z` flattening coefficient.
        f2 (Array):
            The planet's :math:`y` flattening coefficient.
        hotspot_latitude (Array):
            The "latitude" of the hotspot on the planet. Defined the physics way for
            :math:`\\theta` though, not the geography way: 0 is the north pole,
            :math:`\\pi/2` is the equator, and :math:`\\pi` is the south pole.
        hotspot_longitude (Array):
            The longitude of the hotspot on the planet.
        hotspot_concentration (Array):
            The concentration of the hotspot on the planet. :math:`\\kappa` in the
            von Mises-Fisher distribution.

    Returns:
        Array:
            The intensity of the emitted light at each point.
    """
    # always one time slice at a time

    # _uncorrected_emission_profile takes the samples on the planet surface and boosts them onto
    # a unit sphere. Had the samples been uniformly distributed on the planet's surface,
    # applying the correction factor to make up for the squishing would be enough. But,
    # we sampled uniformly on the projected disk, not the 3D surface: they don't occupy
    # an area of 4pi, just pi. So, we divide by another factor of 4 here.

    # To check this, create a super-concentrated hotspot on a tidally locked planet. The
    # peak emission should be nearly equal to emitted_scale when the hotspot faces the
    # observer, since the rest of the contributions are negligible and the area isn't
    # distorted by viewing geometry
    return jnp.sum(
        corrected_emission_profile(
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
        )
    )


@jax.jit
def emission_phase_curve(
    sample_radii,
    sample_thetas,
    two,
    three,
    state,
    **kwargs,
):
    """
    Compute the timeseries of the emitted light from the planet.

    This function does a Monte Carlo estimation of the visible flux emitted by the
    planet at each time step assuming that a) the surface intensity is modeled by a
    von Mises-Fisher distribution and b) the planet is a Lambertian emitter. To save
    on computation, it takes one set of randomly generated samples of a unit disk,
    then coverts these to points on the planet's visible disk at each timestep. This
    introduces some bias- be sure to use a large enough sample it is below an
    appropriate threshold. Also, to compute secondary eclipses, samples are zeroed
    out when they fall behind the star.


    Args:
        sample_radii (Array):
            Randomly sampled radii from a unit sphere, used to generate points on the
            visible disk of the planet at each timestep. Create with
            :func:`generate_sample_radii_thetas`.
        sample_thetas (Array):
            Randomly sampled thetas from a unit sphere, used to generate points on the
            visible disk of the planet at each timestep. Create with
            :func:`generate_sample_radii_thetas`.
        two (dict):
            A dictionary containing the rho coefficients of the planet's implicit 2D
            representation, as seen from the observer and calculated with
            :func:`planet_2d.planet_2d_coeffs`.
        three (dict):
            A dictionary containing the p coefficients of the planet's 3D shape, as seen
            from the observer and calculated with :func:`planet_3d.planet_3d_coeffs`.
        state (dict):
            A dictionary containing all of the parameters needed to compute the phase
            curve. This includes the planet's orbital parameters, the observer's
            parameters, and the hotspot parameters.

    Returns:
        Array:
            The timeseries of observed emitted light from the planet. Each element of
            the array is the total observed emitted flux at that corresponding time in
            the state["times"] array.

    """
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
        em = emission_at_timestep(
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

        return None, em

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
# Stellar effects helpers
########################################################################################


@jax.jit
def stellar_ellipsoidal_variations(true_anomalies, stellar_ellipsoidal_alpha, period):
    """
    Compute the contributions to a phase curve for a star with ellipsoidal variations.

    A simple sinusoid model with minima at primary and secondary eclipse, meant to
    capture gravitational  Works only for a circular orbit and assumes
    that :math:`\\Omega=\\pi`. Uses the model in
    `Shporer et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014ApJ...788...92S/abstract>`_.

    Technically the amplitude in `Morris, Heng, and Kitzmann 2024 <https://ui.adsabs.harvard.edu/abs/2024arXiv240113635M/abstract>`_ is given by

    .. math::

        A_{ellip} = \\frac{\\alpha}{0.077} \\frac{M_p}{M_J} \\left(\\frac{R_s}{R_\\odot}\\right)^3 \\left(\\frac{P}{1 \\text{day}}\\right)^{-2}

    But we instead roll everything into the alpha parameter.

    Args:
        true_anomalies (Array):
            The true anomaly of the planet at each time step.
        stellar_ellipsoidal_alpha (float):
            The amplitude of the ellipsoidal variations.
        period (float):
            The orbital period of the planet.

    Returns:
        Array:
            The contribution to the phase curve from the star's ellipsoidal variations.

    """
    amp = stellar_ellipsoidal_alpha  # / period**2
    phi = true_anomalies - jnp.pi / 2  # orbital phase is zero at primary transit
    phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)
    phi = phi / (2 * jnp.pi)
    return amp * (1 - jnp.cos(4 * jnp.pi * (phi - 0.5)))


@jax.jit
def stellar_doppler_variations(true_anomalies, stellar_doppler_alpha, period):
    """
    Compute the contributions to a phase curve for a star with Doppler variations.

    A simple sinusoid model with a phase of 90 degrees at primary transit meant to
    capture Doppler boosting/flux falling in and out of the bandpass.

    Args:
        true_anomalies (Array):
            The true anomaly of the planet at each time step.
        stellar_doppler_alpha (float):
            The amplitude of the Doppler variations.
        period (float):
            The orbital period of the planet.

    Returns:
        Array:
            The contribution to the phase curve from the star's Doppler variations.
    """

    amp = stellar_doppler_alpha  # / period**(-1/3)
    phi = true_anomalies - jnp.pi / 2  # orbital phase is zero at primary transit
    phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)
    phi = phi / (2 * jnp.pi)
    return amp * jnp.sin(2 * jnp.pi * phi)


########################################################################################
# Combined curves
########################################################################################


@jax.jit
def phase_curve(sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c):
    """
    Compute the reflected and emitted phase curves of the planet.

    This is essentially a wrapper for :func:`reflected_phase_curve` and
    :func:`emission_phase_curve`, except is reuses computations where it can, and also
    applies the appriate scalings to each (albedo/distance from star/area seen by star
    for reflection, and the emitted scale for emission).

    Args:
        sample_radii (Array):
            Randomly sampled radii from a unit sphere, used to generate points on the
            visible disk of the planet at each timestep. Create with
            :func:`generate_sample_radii_thetas`.
        sample_thetas (Array):
            Randomly sampled thetas from a unit sphere, used to generate points on the
            visible disk of the planet at each timestep. Create with
            :func:`generate_sample_radii_thetas`.
        two (dict):
            A dictionary containing the rho coefficients of the planet's implicit 2D
            representation, as seen from the observer and calculated with
            :func:`planet_2d.planet_2d_coeffs`.
        three (dict):
            A dictionary containing the p coefficients of the planet's 3D shape, as seen
            from the observer and calculated with :func:`planet_3d.planet_3d_coeffs`.
        state (dict):
            A dictionary containing all of the parameters needed to compute the phase
            curve. This includes the planet's orbital parameters, the observer's
            parameters, and the hotspot parameters.
        x_c (Array): The x coordinate of the center of the planet.
        y_c (Array): The y coordinate of the center of the planet.
        z_c (Array): The z coordinate of the center of the planet.

    Returns:
        Tuple:
            The correctly scaled reflected and emitted contributions to the phase curve.

    """

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

        mask = ~(((x) ** 2 + (y) ** 2 < 1) & ((z) < 0))
        lamb = lamb * mask

        # emission stuff
        em = emission_at_timestep(
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

        return None, (
            jnp.sum(lamb) / sample_radii.shape[0],
            em,
        )

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

    reflected_norm = reflected_normalization(two, three, x_c, y_c, z_c)

    return (
        fluxes[0] * reflected_norm * state["albedo"],
        fluxes[1] * state["emitted_scale"],
    )  # the reflected and emitted contributions
