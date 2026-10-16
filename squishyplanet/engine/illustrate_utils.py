import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from squishyplanet.engine.parametric_ellipse import poly_to_parametric_helper


@jax.jit
def generate_sample_radii_thetas(
    key: jax.Array, num_points: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Create a random set of radii and thetas for sampling the planet's surface.

    These are uniformly distributed through a unit circle and will be scaled and rotated
    to match the planet's shape and orientation at each timestep. However, they will
    be reused at every time step, which could introduce a bias but makes things much
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
    sample_radii: jax.Array,
    sample_thetas: jax.Array,
    rho_xx: jax.Array,
    rho_xy: jax.Array,
    rho_x0: jax.Array,
    rho_yy: jax.Array,
    rho_y0: jax.Array,
    rho_00: jax.Array,
    **kwargs: object,
) -> tuple[jax.Array, jax.Array]:
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
    x: jax.Array,
    y: jax.Array,
    p_xx: jax.Array,
    p_xy: jax.Array,
    p_xz: jax.Array,
    p_x0: jax.Array,
    p_yy: jax.Array,
    p_yz: jax.Array,
    p_y0: jax.Array,
    p_zz: jax.Array,
    p_z0: jax.Array,
    p_00: jax.Array,
    **kwargs: object,
) -> jax.Array:
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
    sample_radii: jax.Array,
    sample_thetas: jax.Array,
    rho_xx: jax.Array,
    rho_xy: jax.Array,
    rho_x0: jax.Array,
    rho_yy: jax.Array,
    rho_y0: jax.Array,
    rho_00: jax.Array,
    p_xx: jax.Array,
    p_xy: jax.Array,
    p_xz: jax.Array,
    p_x0: jax.Array,
    p_yy: jax.Array,
    p_yz: jax.Array,
    p_y0: jax.Array,
    p_zz: jax.Array,
    p_z0: jax.Array,
    p_00: jax.Array,
    **kwargs: object,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert randomly sampled :math:`(x, y)` points on the projected planet to
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
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    p_xx: jax.Array,
    p_xy: jax.Array,
    p_xz: jax.Array,
    p_x0: jax.Array,
    p_yy: jax.Array,
    p_yz: jax.Array,
    p_y0: jax.Array,
    p_zz: jax.Array,
    p_z0: jax.Array,
    p_00: jax.Array,
) -> jax.Array:
    """Compute the unit normal vector to the planet's surface at a given point.

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
    planet_surface_normal: jax.Array,
    x_c: jax.Array,
    y_c: jax.Array,
    z_c: jax.Array,
    **kwargs: object,
) -> jax.Array:
    """A helper function to compute the cosine of the angle between the planet's surface
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
def lambertian_reflection(
    surface_star_cos_angle: jax.Array, x: jax.Array, y: jax.Array, z: jax.Array
) -> jax.Array:
    """Compute the reflected intensity at a specific point on the planet's surface assuming
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
