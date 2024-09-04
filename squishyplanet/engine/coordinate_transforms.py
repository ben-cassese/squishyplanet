import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def create_sky_to_planet_transform(a, e, f, Omega, i, omega, r, obliq, prec, **kwargs):
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
def create_planet_to_sky_transform(a, e, f, Omega, i, omega, r, obliq, prec, **kwargs):
    """
    Compute the rotation matrix to go from the planet's frame to the sky.

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

    def _x_x(a, e, f, Omega, i, omega, r, phi, theta):
        return (
            -(jnp.sin(i) * jnp.sin(phi) * jnp.sin(Omega))
            - jnp.cos(phi)
            * jnp.sin(theta)
            * (
                jnp.cos(Omega) * jnp.sin(omega)
                + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
            )
            + jnp.cos(theta)
            * jnp.cos(phi)
            * (
                jnp.cos(omega) * jnp.cos(Omega)
                - jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
            )
        )

    def _x_y(a, e, f, Omega, i, omega, r, phi, theta):
        return -(
            jnp.cos(omega)
            * (
                jnp.cos(Omega) * jnp.sin(theta)
                + jnp.cos(i) * jnp.cos(theta) * jnp.sin(Omega)
            )
        ) + jnp.sin(omega) * (
            -(jnp.cos(theta) * jnp.cos(Omega))
            + jnp.cos(i) * jnp.sin(theta) * jnp.sin(Omega)
        )

    def _x_z(a, e, f, Omega, i, omega, r, phi, theta):
        return (
            -(jnp.cos(Omega) * jnp.sin(theta) * jnp.sin(phi) * jnp.sin(omega))
            + (
                jnp.cos(phi) * jnp.sin(i)
                - jnp.cos(i) * jnp.cos(omega) * jnp.sin(theta) * jnp.sin(phi)
            )
            * jnp.sin(Omega)
            + jnp.cos(theta)
            * jnp.sin(phi)
            * (
                jnp.cos(omega) * jnp.cos(Omega)
                - jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
            )
        )

    def _x_0(a, e, f, Omega, i, omega, r, phi, theta):
        return (
            a
            * (-1 + e**2)
            * (
                jnp.sin(f)
                * (
                    jnp.cos(Omega) * jnp.sin(omega)
                    + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
                )
                + jnp.cos(f)
                * (
                    -(jnp.cos(omega) * jnp.cos(Omega))
                    + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
                )
            )
        ) / (1 + e * jnp.cos(f))

    def _y_x(a, e, f, Omega, i, omega, r, phi, theta):
        return jnp.cos(Omega) * (
            jnp.sin(i) * jnp.sin(phi)
            + jnp.cos(i) * jnp.cos(phi) * jnp.sin(theta + omega)
        ) + jnp.cos(phi) * jnp.cos(theta + omega) * jnp.sin(Omega)

    def _y_y(a, e, f, Omega, i, omega, r, phi, theta):
        return jnp.cos(i) * jnp.cos(theta + omega) * jnp.cos(Omega) - jnp.sin(
            theta + omega
        ) * jnp.sin(Omega)

    def _y_z(a, e, f, Omega, i, omega, r, phi, theta):
        return -(jnp.cos(phi) * jnp.cos(Omega) * jnp.sin(i)) + jnp.sin(phi) * (
            jnp.cos(i) * jnp.cos(Omega) * jnp.sin(theta + omega)
            + jnp.cos(theta + omega) * jnp.sin(Omega)
        )

    def _y_0(a, e, f, Omega, i, omega, r, phi, theta):
        return -(
            (
                a
                * (-1 + e**2)
                * (
                    jnp.cos(i) * jnp.cos(Omega) * jnp.sin(f + omega)
                    + jnp.cos(f + omega) * jnp.sin(Omega)
                )
            )
            / (1 + e * jnp.cos(f))
        )

    def _z_x(a, e, f, Omega, i, omega, r, phi, theta):
        return -(jnp.cos(i) * jnp.sin(phi)) + jnp.cos(phi) * jnp.sin(i) * jnp.sin(
            theta + omega
        )

    def _z_y(a, e, f, Omega, i, omega, r, phi, theta):
        return jnp.cos(theta + omega) * jnp.sin(i)

    def _z_z(a, e, f, Omega, i, omega, r, phi, theta):
        return jnp.cos(i) * jnp.cos(phi) + jnp.sin(i) * jnp.sin(phi) * jnp.sin(
            theta + omega
        )

    def _z_0(a, e, f, Omega, i, omega, r, phi, theta):
        return -(
            (a * (-1 + e**2) * jnp.sin(i) * jnp.sin(f + omega)) / (1 + e * jnp.cos(f))
        )

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
