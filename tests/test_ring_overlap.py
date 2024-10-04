import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def sample_points_in_unit_disk(num_points):
    """Uniformly sample points from the unit disk."""
    angles = jnp.linspace(0, 2 * jnp.pi, num_points)
    radii = jnp.sqrt(jnp.random.uniform(0, 1, num_points))
    xs = radii * jnp.cos(angles)
    ys = radii * jnp.sin(angles)
    return xs, ys


@jax.jit
def is_within_ellipse(rho_coeffs, xs, ys):
    """Check if the points (xs, ys) are within the ellipse defined by the polynomial coefficients."""
    return (
        rho_coeffs["rho_xx"] * xs**2
        + rho_coeffs["rho_xy"] * xs * ys
        + rho_coeffs["rho_x0"] * xs
        + rho_coeffs["rho_yy"] * ys**2
        + rho_coeffs["rho_y0"] * ys
        + rho_coeffs["rho_00"]
    ) < 1


@jax.jit
def measure_overlap_area(rho_coeffs1, rho_coeffs2, num_samples=10000):
    """Measure the area of overlap between two ellipses within the unit disk."""
    xs, ys = sample_points_in_unit_disk(num_samples)
    within_ellipse1 = is_within_ellipse(rho_coeffs1, xs, ys)
    within_ellipse2 = is_within_ellipse(rho_coeffs2, xs, ys)
    overlap_count = jnp.sum(within_ellipse1 & within_ellipse2)
    area_unit_disk = jnp.pi
    overlap_area = (overlap_count / num_samples) * area_unit_disk
    return overlap_area
