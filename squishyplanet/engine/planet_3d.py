import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from squishyplanet.engine.kepler import skypos


def _p_xx(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        (
            jnp.cos(omega)
            * (
                jnp.cos(Omega) * jnp.sin(prec)
                + jnp.cos(i) * jnp.cos(prec) * jnp.sin(Omega)
            )
            + jnp.sin(omega)
            * (
                jnp.cos(prec) * jnp.cos(Omega)
                - jnp.cos(i) * jnp.sin(prec) * jnp.sin(Omega)
            )
        )
        ** 2
        / (-1 + f2) ** 2
        + (
            jnp.sin(i) * jnp.sin(obliq) * jnp.sin(Omega)
            + jnp.cos(obliq)
            * jnp.sin(prec)
            * (
                jnp.cos(Omega) * jnp.sin(omega)
                + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
            )
            + jnp.cos(prec)
            * jnp.cos(obliq)
            * (
                -(jnp.cos(omega) * jnp.cos(Omega))
                + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
            )
        )
        ** 2
        + (
            jnp.cos(Omega) * jnp.sin(prec) * jnp.sin(obliq) * jnp.sin(omega)
            + (
                -(jnp.cos(obliq) * jnp.sin(i))
                + jnp.cos(i) * jnp.cos(omega) * jnp.sin(prec) * jnp.sin(obliq)
            )
            * jnp.sin(Omega)
            + jnp.cos(prec)
            * jnp.sin(obliq)
            * (
                -(jnp.cos(omega) * jnp.cos(Omega))
                + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
            )
        )
        ** 2
        / (-1 + f1) ** 2
    ) / r**2


def _p_xy(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        (
            16
            * (
                jnp.cos(Omega) ** 2
                * jnp.sin(i)
                * jnp.sin(prec)
                * jnp.sin(2 * obliq)
                * jnp.sin(omega)
                - 2
                * jnp.cos(obliq) ** 2
                * jnp.cos(Omega)
                * jnp.sin(i) ** 2
                * jnp.sin(Omega)
                - 2
                * jnp.cos(obliq)
                * jnp.sin(i)
                * jnp.sin(prec)
                * jnp.sin(obliq)
                * jnp.sin(omega)
                * jnp.sin(Omega) ** 2
                + jnp.cos(i)
                * jnp.sin(obliq)
                * (
                    jnp.cos(2 * omega)
                    * jnp.cos(2 * Omega)
                    * jnp.sin(2 * prec)
                    * jnp.sin(obliq)
                    + jnp.cos(prec) ** 2
                    * jnp.cos(2 * Omega)
                    * jnp.sin(obliq)
                    * jnp.sin(2 * omega)
                    - jnp.cos(2 * Omega)
                    * jnp.sin(prec) ** 2
                    * jnp.sin(obliq)
                    * jnp.sin(2 * omega)
                    + 4
                    * jnp.cos(obliq)
                    * jnp.cos(omega)
                    * jnp.cos(Omega)
                    * jnp.sin(i)
                    * jnp.sin(prec)
                    * jnp.sin(Omega)
                    + 4
                    * jnp.cos(prec)
                    * jnp.cos(obliq)
                    * jnp.cos(Omega)
                    * jnp.sin(i)
                    * jnp.sin(omega)
                    * jnp.sin(Omega)
                )
                - jnp.cos(prec)
                * jnp.cos(omega)
                * (
                    jnp.cos(2 * Omega) * jnp.sin(i) * jnp.sin(2 * obliq)
                    + 4
                    * jnp.cos(Omega)
                    * jnp.sin(prec)
                    * jnp.sin(obliq) ** 2
                    * jnp.sin(omega)
                    * jnp.sin(Omega)
                )
                + jnp.cos(prec) ** 2
                * jnp.cos(omega) ** 2
                * jnp.sin(obliq) ** 2
                * jnp.sin(2 * Omega)
                + jnp.sin(prec) ** 2
                * jnp.sin(obliq) ** 2
                * jnp.sin(omega) ** 2
                * jnp.sin(2 * Omega)
                - jnp.cos(i) ** 2
                * jnp.sin(obliq) ** 2
                * jnp.sin(prec + omega) ** 2
                * jnp.sin(2 * Omega)
            )
        )
        / (-1 + f1) ** 2
        - 32
        * (
            -(
                jnp.cos(prec) ** 2
                * jnp.cos(obliq) ** 2
                * jnp.cos(omega) ** 2
                * jnp.cos(Omega)
                * jnp.sin(Omega)
            )
            + jnp.cos(Omega) * jnp.sin(i) ** 2 * jnp.sin(obliq) ** 2 * jnp.sin(Omega)
            + jnp.cos(prec)
            * jnp.cos(obliq)
            * jnp.cos(omega)
            * (
                -(jnp.cos(Omega) ** 2 * jnp.sin(i) * jnp.sin(obliq))
                - jnp.cos(i)
                * jnp.cos(obliq)
                * jnp.cos(2 * Omega)
                * jnp.sin(prec + omega)
                + jnp.sin(i) * jnp.sin(obliq) * jnp.sin(Omega) ** 2
                + jnp.cos(obliq) * jnp.sin(prec) * jnp.sin(omega) * jnp.sin(2 * Omega)
            )
            + jnp.cos(obliq)
            * jnp.sin(i)
            * jnp.sin(obliq)
            * (
                jnp.cos(Omega) ** 2 * jnp.sin(prec) * jnp.sin(omega)
                - jnp.sin(prec) * jnp.sin(omega) * jnp.sin(Omega) ** 2
                + jnp.cos(i) * jnp.sin(prec + omega) * jnp.sin(2 * Omega)
            )
            + jnp.cos(obliq) ** 2
            * (
                jnp.cos(i)
                * jnp.cos(2 * Omega)
                * jnp.sin(prec)
                * jnp.sin(omega)
                * jnp.sin(prec + omega)
                - jnp.cos(Omega)
                * jnp.sin(prec) ** 2
                * jnp.sin(omega) ** 2
                * jnp.sin(Omega)
                + (jnp.cos(i) ** 2 * jnp.sin(prec + omega) ** 2 * jnp.sin(2 * Omega))
                / 2.0
            )
        )
        + (
            4 * jnp.sin(i - 2 * (prec + omega - Omega))
            - 4 * jnp.sin(i + 2 * (prec + omega - Omega))
            + 2 * jnp.sin(2 * (i - Omega))
            + jnp.sin(2 * (i - prec - omega - Omega))
            + 6 * jnp.sin(2 * (prec + omega - Omega))
            + jnp.sin(2 * (i + prec + omega - Omega))
            + 4 * jnp.sin(2 * Omega)
            - 2 * jnp.sin(2 * (i + Omega))
            - jnp.sin(2 * (i - prec - omega + Omega))
            - 6 * jnp.sin(2 * (prec + omega + Omega))
            - jnp.sin(2 * (i + prec + omega + Omega))
            + 4 * jnp.sin(i - 2 * (prec + omega + Omega))
            - 4 * jnp.sin(i + 2 * (prec + omega + Omega))
        )
        / (-1 + f2) ** 2
    ) / (16.0 * r**2)


def _p_xz(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        2
        * (
            -(
                (
                    jnp.cos(prec + omega)
                    * jnp.sin(i)
                    * (
                        jnp.cos(omega)
                        * (
                            jnp.cos(Omega) * jnp.sin(prec)
                            + jnp.cos(i) * jnp.cos(prec) * jnp.sin(Omega)
                        )
                        + jnp.sin(omega)
                        * (
                            jnp.cos(prec) * jnp.cos(Omega)
                            - jnp.cos(i) * jnp.sin(prec) * jnp.sin(Omega)
                        )
                    )
                )
                / (-1 + f2) ** 2
            )
            + (
                jnp.cos(i) * jnp.sin(obliq)
                - jnp.cos(obliq) * jnp.sin(i) * jnp.sin(prec + omega)
            )
            * (
                jnp.sin(i) * jnp.sin(obliq) * jnp.sin(Omega)
                + jnp.cos(obliq)
                * jnp.sin(prec)
                * (
                    jnp.cos(Omega) * jnp.sin(omega)
                    + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
                )
                + jnp.cos(prec)
                * jnp.cos(obliq)
                * (
                    -(jnp.cos(omega) * jnp.cos(Omega))
                    + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
                )
            )
            - (
                (
                    jnp.cos(i) * jnp.cos(obliq)
                    + jnp.sin(i) * jnp.sin(obliq) * jnp.sin(prec + omega)
                )
                * (
                    jnp.cos(Omega) * jnp.sin(prec) * jnp.sin(obliq) * jnp.sin(omega)
                    + (
                        -(jnp.cos(obliq) * jnp.sin(i))
                        + jnp.cos(i) * jnp.cos(omega) * jnp.sin(prec) * jnp.sin(obliq)
                    )
                    * jnp.sin(Omega)
                    + jnp.cos(prec)
                    * jnp.sin(obliq)
                    * (
                        -(jnp.cos(omega) * jnp.cos(Omega))
                        + jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
                    )
                )
            )
            / (-1 + f1) ** 2
        )
    ) / r**2


def _p_x0(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        2
        * a
        * (-1 + e**2)
        * (
            -(
                (
                    jnp.sin(f - prec)
                    * (
                        jnp.cos(omega)
                        * (
                            jnp.cos(Omega) * jnp.sin(prec)
                            + jnp.cos(i) * jnp.cos(prec) * jnp.sin(Omega)
                        )
                        + jnp.sin(omega)
                        * (
                            jnp.cos(prec) * jnp.cos(Omega)
                            - jnp.cos(i) * jnp.sin(prec) * jnp.sin(Omega)
                        )
                    )
                )
                / (-1 + f2) ** 2
            )
            + (
                jnp.cos(f - prec)
                * (
                    jnp.cos(prec)
                    * (2 - 2 * f1 + f1**2 + (-2 + f1) * f1 * jnp.cos(2 * obliq))
                    * (
                        jnp.cos(omega) * jnp.cos(Omega)
                        - jnp.cos(i) * jnp.sin(omega) * jnp.sin(Omega)
                    )
                    - 2
                    * (
                        (-2 + f1)
                        * f1
                        * jnp.cos(obliq)
                        * jnp.sin(i)
                        * jnp.sin(obliq)
                        * jnp.sin(Omega)
                        + (-1 + f1) ** 2
                        * jnp.cos(obliq) ** 2
                        * jnp.sin(prec)
                        * (
                            jnp.cos(Omega) * jnp.sin(omega)
                            + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
                        )
                        + jnp.sin(prec)
                        * jnp.sin(obliq) ** 2
                        * (
                            jnp.cos(Omega) * jnp.sin(omega)
                            + jnp.cos(i) * jnp.cos(omega) * jnp.sin(Omega)
                        )
                    )
                )
            )
            / (2.0 * (-1 + f1) ** 2)
        )
    ) / (r**2 * (1 + e * jnp.cos(f)))


def _p_yy(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        (
            jnp.cos(Omega)
            * (
                jnp.sin(i) * jnp.sin(obliq)
                + jnp.cos(i) * jnp.cos(obliq) * jnp.sin(prec + omega)
            )
            + jnp.cos(obliq) * jnp.cos(prec + omega) * jnp.sin(Omega)
        )
        ** 2
        + (
            jnp.cos(i) * jnp.cos(prec + omega) * jnp.cos(Omega)
            - jnp.sin(prec + omega) * jnp.sin(Omega)
        )
        ** 2
        / (-1 + f2) ** 2
        + (
            jnp.cos(obliq) * jnp.cos(Omega) * jnp.sin(i)
            - jnp.sin(obliq)
            * (
                jnp.cos(i) * jnp.cos(Omega) * jnp.sin(prec + omega)
                + jnp.cos(prec + omega) * jnp.sin(Omega)
            )
        )
        ** 2
        / (-1 + f1) ** 2
    ) / r**2


def _p_yz(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        2
        * (
            -(
                (
                    jnp.cos(i) * jnp.sin(obliq)
                    - jnp.cos(obliq) * jnp.sin(i) * jnp.sin(prec + omega)
                )
                * (
                    jnp.cos(Omega)
                    * (
                        jnp.sin(i) * jnp.sin(obliq)
                        + jnp.cos(i) * jnp.cos(obliq) * jnp.sin(prec + omega)
                    )
                    + jnp.cos(obliq) * jnp.cos(prec + omega) * jnp.sin(Omega)
                )
            )
            + (
                jnp.cos(prec + omega)
                * jnp.sin(i)
                * (
                    jnp.cos(i) * jnp.cos(prec + omega) * jnp.cos(Omega)
                    - jnp.sin(prec + omega) * jnp.sin(Omega)
                )
            )
            / (-1 + f2) ** 2
            + (
                (
                    jnp.cos(i) * jnp.cos(obliq)
                    + jnp.sin(i) * jnp.sin(obliq) * jnp.sin(prec + omega)
                )
                * (
                    -(jnp.cos(obliq) * jnp.cos(Omega) * jnp.sin(i))
                    + jnp.sin(obliq)
                    * (
                        jnp.cos(i) * jnp.cos(Omega) * jnp.sin(prec + omega)
                        + jnp.cos(prec + omega) * jnp.sin(Omega)
                    )
                )
            )
            / (-1 + f1) ** 2
        )
    ) / r**2


def _p_y0(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        2
        * a
        * (-1 + e**2)
        * (
            jnp.cos(f - prec)
            * jnp.cos(obliq)
            * (
                jnp.cos(Omega)
                * (
                    jnp.sin(i) * jnp.sin(obliq)
                    + jnp.cos(i) * jnp.cos(obliq) * jnp.sin(prec + omega)
                )
                + jnp.cos(obliq) * jnp.cos(prec + omega) * jnp.sin(Omega)
            )
            + (
                jnp.sin(f - prec)
                * (
                    jnp.cos(i) * jnp.cos(prec + omega) * jnp.cos(Omega)
                    - jnp.sin(prec + omega) * jnp.sin(Omega)
                )
            )
            / (-1 + f2) ** 2
            + (
                jnp.cos(f - prec)
                * jnp.sin(obliq)
                * (
                    -(jnp.cos(obliq) * jnp.cos(Omega) * jnp.sin(i))
                    + jnp.sin(obliq)
                    * (
                        jnp.cos(i) * jnp.cos(Omega) * jnp.sin(prec + omega)
                        + jnp.cos(prec + omega) * jnp.sin(Omega)
                    )
                )
            )
            / (-1 + f1) ** 2
        )
    ) / (r**2 * (1 + e * jnp.cos(f)))


def _p_zz(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        (jnp.cos(prec + omega) ** 2 * jnp.sin(i) ** 2) / (-1 + f2) ** 2
        + (
            jnp.cos(i) * jnp.sin(obliq)
            - jnp.cos(obliq) * jnp.sin(i) * jnp.sin(prec + omega)
        )
        ** 2
        + (
            jnp.cos(i) * jnp.cos(obliq)
            + jnp.sin(i) * jnp.sin(obliq) * jnp.sin(prec + omega)
        )
        ** 2
        / (-1 + f1) ** 2
    ) / r**2


def _p_z0(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        2
        * a
        * (-1 + e**2)
        * (
            (jnp.cos(prec + omega) * jnp.sin(i) * jnp.sin(f - prec)) / (-1 + f2) ** 2
            + (
                jnp.cos(f - prec)
                * (
                    -((-2 + f1) * f1 * jnp.cos(i) * jnp.cos(obliq) * jnp.sin(obliq))
                    + jnp.sin(i)
                    * ((-1 + f1) ** 2 * jnp.cos(obliq) ** 2 + jnp.sin(obliq) ** 2)
                    * jnp.sin(prec + omega)
                )
            )
            / (-1 + f1) ** 2
        )
    ) / (r**2 * (1 + e * jnp.cos(f)))


def _p_00(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2):
    return (
        a**2
        * (-1 + e**2) ** 2
        * (
            jnp.sin(f - prec) ** 2 / (-1 + f2) ** 2
            + (
                jnp.cos(f - prec) ** 2
                * ((-1 + f1) ** 2 * jnp.cos(obliq) ** 2 + jnp.sin(obliq) ** 2)
            )
            / (-1 + f1) ** 2
        )
    ) / (r + e * r * jnp.cos(f)) ** 2


@jax.jit
def planet_3d_coeffs(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2, **kwargs):
    """
    Calculate and return the coefficients that describe the planet as an implicit
    surface in 3D space as a function of its orbital state.

    This function computes a dictionary of coefficients related to the 3D position
    and orientation of a planet given its orbital and rotational characteristics. All
    inputs must be jnp.ndarrays. They can either be shape (1,) or (N,) with 1 unique
    N allowed per call (e.g., everything but f and prec are single-valued, but those
    two are length N).

    Args:
        a (Array [Rstar]):
            Semi-major axis of the orbit in units of R_star
        e (Array [Dimensionless]):
            Eccentricity of the orbit
        f (Array [Radian]):
            True anomaly, the angle between the direction of periapsis and the current
            position of the planet as seen from the star.
        Omega (Array [Radian]):
            Longitude of the ascending node
        i (Array [Radian]):
            Orbital inclination
        omega (Array [Radian]):
            Argument of periapsis
        r (Array [Rstar]):
            Equatorial radius of the planet, in units of R_star
        obliq (Array [Radian]):
            Obliquity, the angle between the planet's orbital plane and its equatorial
            plane. Defined when the planet is at periapsis with an Omega of zero as a
            rotation around the sky-frame y-axis, such that a positive obliquity tips
            the planet's north pole away from the star.
        prec (Array [Radian]):
            Precession angle, or equivalently the longitude of ascending node of the
            planet's equatorial plane. This is defined at periapsis with an Omega of
            zero as a rotation about the sky-frame z-axis.
        f1 (Array [Dimensionless]):
            The flattening coefficient of the planet that describes the compression
            along the planet's polar axis. A value of 0.0 indicates no flattening.
        f2 (Array [Dimensionless]):
            The flattening coefficient of the planet that describes the compression
            along the planet's "y" axis, the vector in its equatorial plane that is
            perpendicular to the direction of motion at periapsis, assuming the 0.0
            obliquity.
        **kwargs:
            Unused additional keyword arguments. These are included so that we can can
            take in a larger state dictionary that includes all of the required
            parameters along with other unnecessary ones.

    Returns:
        dict:
            A dictionary with keys representing different coefficient names and
            their corresponding values. The coeffients satisfy the implicit equation:

            .. math::
                p_{xx} x^2 + p_{xy} xy + p_{xz} xz + p_{x0} x + p_{yy} y^2 + p_{yz} yz + p_{y0} y + p_{zz} z^2 + p_{z0} z + p_{00} = 1
    """
    return {
        "p_xx": _p_xx(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
        "p_xy": _p_xy(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
        "p_xz": _p_xz(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
        "p_x0": _p_x0(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
        "p_yy": _p_yy(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
        "p_yz": _p_yz(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
        "p_y0": _p_y0(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
        "p_zz": _p_zz(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
        "p_z0": _p_z0(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
        "p_00": _p_00(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2),
    }


def extended_illumination_offsets(
    a, e, f, Omega, i, omega, extended_illumination_points, **kwargs
):
    """
    Generate a set of points uniformly distributed on the portion of the star visible
    from the planet.

    This takes a spherical cap of points centered on the north pole of the star,
    squishes them together so they'd all be visible from the planet, and then rotates
    them to be centered on the sub-planet point on the star.

    Args:
        a (Array [Rstar]):
            Semi-major axis of the orbit in units of R_star
        e (Array [Dimensionless]):
            Eccentricity of the orbit
        f (Array [Radian]):
            True anomaly, the angle between the direction of periapsis and the current
            position of the planet as seen from the star.
        Omega (Array [Radian]):
            Longitude of the ascending node
        i (Array [Radian]):
            Orbital inclination
        omega (Array [Radian]):
            Argument of periapsis
        extended_illumination_points (Array):
            A set of points lying on a unit hemisphere centered at the origin that are
            evenly distributed across the projected disk when viewed from above. Created
            in during initialization of an :class:`OblateSystem` object.
        **kwargs:
            Unused additional keyword arguments. These are included so that we can can
            take in a larger state dictionary that includes all of the required
            parameters along with other unnecessary ones.

    Returns:
        Array:
            A set of points on the star that are visible from the planet, evenly
            distributed across the projected disk of the star as seen by the planet, and
            centered on the sub-planet point on the star.

    """

    xc, yc, zc = skypos(a, e, f, Omega, i, omega)

    # account for the fact that an observer at the center of the planet couldn't see
    # the whole star: the effective disk is shrunk
    r = jnp.linalg.norm(jnp.array([xc, yc, zc]))
    r_eff = jnp.sqrt(-1 + r**2) / r
    extended_illumination_points = extended_illumination_points.at[:, 0].set(
        r_eff * extended_illumination_points[:, 0]
    )
    extended_illumination_points = extended_illumination_points.at[:, 1].set(
        r_eff * extended_illumination_points[:, 1]
    )
    extended_illumination_points = extended_illumination_points.at[:, 2].set(
        jnp.sqrt(
            1
            - extended_illumination_points[:, 0] ** 2
            - extended_illumination_points[:, 1] ** 2
        )
    )

    # rotate those points to be from the perspective of an observer at the center of the
    # planet
    x = jnp.array([xc, yc, zc])
    x = x / jnp.linalg.norm(x, axis=0)
    thetas = jnp.arccos(x[-1])
    phis = jnp.arctan2(x[0], x[1])

    rot_x = lambda theta: jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(theta), -jnp.sin(theta)],
            [0, jnp.sin(theta), jnp.cos(theta)],
        ]
    )
    rot_z = lambda phi: jnp.array(
        [[jnp.cos(phi), -jnp.sin(phi), 0], [jnp.sin(phi), jnp.cos(phi), 0], [0, 0, 1]]
    )

    rotate_pt = lambda theta, phi, pt: jnp.dot(rot_z(phi), jnp.dot(rot_x(theta), pt))
    func = lambda pt: jax.vmap(rotate_pt, in_axes=(0, 0, None))(thetas, phis, pt)
    pts = jax.vmap(func)(extended_illumination_points)
    return pts


@jax.jit
def planet_3d_coeffs_extended_illumination(
    a,
    e,
    f,
    Omega,
    i,
    omega,
    r,
    obliq,
    prec,
    f1,
    f2,
    offsets,
    **kwargs,
):
    """
    Generate many sets of p coefficients that describe same planet offset from its
    true position by different amounts.

    Since the star is not actually a point source, we slightly underestimate the
    area of the illuminated portion of the planet. The limb of the star can "see around
    the horizon", and this extra illumination will affect the reflected portion of a
    phase curve. To crudely account for this, we can break the star into many point
    sources distributed over the portion of the star that is visible from the planet,
    then add their resulting lightcurves. This isn't perfect for a few reasons: how
    should we distribute this point sources, and how should we weight them? Also, for
    a non-spherical planet, what do we mean by "the portion of the star that is visible
    from the planet"? For now, we avoid those questions by assigning equal intensities
    to a set of points distributed uniformly over the portion of the hemisphere of the
    star that would be visible to an observer at the center of the planet.

    Args:
        a (Array [Rstar]):
            Semi-major axis of the orbit in units of R_star
        e (Array [Dimensionless]):
            Eccentricity of the orbit
        f (Array [Radian]):
            True anomaly, the angle between the direction of periapsis and the current
            position of the planet as seen from the star.
        Omega (Array [Radian]):
            Longitude of the ascending node
        i (Array [Radian]):
            Orbital inclination
        omega (Array [Radian]):
            Argument of periapsis
        r (Array [Rstar]):
            Equatorial radius of the planet, in units of R_star
        obliq (Array [Radian]):
            Obliquity, the angle between the planet's orbital plane and its equatorial
            plane. Defined when the planet is at periapsis with an Omega of zero as a
            rotation around the sky-frame y-axis, such that a positive obliquity tips
            the planet's north pole away from the star.
        prec (Array [Radian]):
            Precession angle, or equivalently the longitude of ascending node of the
            planet's equatorial plane. This is defined at periapsis with an Omega of
            zero as a rotation about the sky-frame z-axis.
        f1 (Array [Dimensionless]):
            The flattening coefficient of the planet that describes the compression
            along the planet's polar axis. A value of 0.0 indicates no flattening.
        f2 (Array [Dimensionless]):
            The flattening coefficient of the planet that describes the compression
            along the planet's "y" axis, the vector in its equatorial plane that is
            perpendicular to the direction of motion at periapsis, assuming the 0.0
            obliquity.
        offsets (Array [Rstar]):
            An array of offsets from the planet's true position. Each offset is a
            3-element array representing the x, y, and z offsets from the planet's
            true position in units of R_star. Used when splitting the star into many
            point sources to account for extended illumination.
        **kwargs:
            Unused additional keyword arguments. These are included so that we can can
            take in a larger state dictionary that includes all of the required
            parameters along with other unnecessary ones.

    Returns:
        dict:
            A dictionary similar to the one returned by :func:`planet_3d_coeffs`, but
            with now describing the planet after being translated by the provided
            offsets.

    """

    if prec.shape != f.shape:
        prec = jnp.ones_like(f) * prec

    unshifted = planet_3d_coeffs(a, e, f, Omega, i, omega, r, obliq, prec, f1, f2)
    p_xx = unshifted["p_xx"][None, :]
    p_xy = unshifted["p_xy"][None, :]
    p_xz = unshifted["p_xz"][None, :]
    p_x0 = unshifted["p_x0"][None, :]
    p_yy = unshifted["p_yy"][None, :]
    p_yz = unshifted["p_yz"][None, :]
    p_y0 = unshifted["p_y0"][None, :]
    p_zz = unshifted["p_zz"][None, :]
    p_z0 = unshifted["p_z0"][None, :]
    p_00 = unshifted["p_00"][None, :]

    # surface_pt_index, f_index, xyz_index
    xo = offsets[..., 0]
    yo = offsets[..., 1]
    zo = offsets[..., 2]

    # CoefficientRules[(pxx x^2 + pxy x y + pxz x z + px0 x + pyy y^2 +
    # pyz y z + py0 y + pzz z^2 + pz0 z + p00 /. {x -> x - xo,
    # y -> y - yo, z -> z - zo}), {x, y, z}]
    return {
        "p_xx": jnp.ones_like(xo) * p_xx,
        "p_xy": jnp.ones_like(xo) * p_xy,
        "p_xz": jnp.ones_like(xo) * p_xz,
        "p_x0": p_x0 - 2 * p_xx * xo - p_xy * yo - p_xz * zo,
        "p_yy": jnp.ones_like(xo) * p_yy,
        "p_yz": jnp.ones_like(xo) * p_yz,
        "p_y0": p_y0 - p_xy * xo - 2 * p_yy * yo - p_yz * zo,
        "p_zz": jnp.ones_like(xo) * p_zz,
        "p_z0": p_z0 - p_xz * xo - p_yz * yo - 2 * p_zz * zo,
        "p_00": p_00
        - p_x0 * xo
        + p_xx * xo**2
        - p_y0 * yo
        + p_xy * xo * yo
        + p_yy * yo**2
        - p_z0 * zo
        + p_xz * xo * zo
        + p_yz * yo * zo
        + p_zz * zo**2,
    }
