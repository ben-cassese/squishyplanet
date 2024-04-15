import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


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
            + jnp.cos(Omega)
            * jnp.sin(i) ** 2
            * jnp.sin(obliq) ** 2
            * jnp.sin(Omega)
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
                + jnp.cos(obliq)
                * jnp.sin(prec)
                * jnp.sin(omega)
                * jnp.sin(2 * Omega)
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
                + (
                    jnp.cos(i) ** 2
                    * jnp.sin(prec + omega) ** 2
                    * jnp.sin(2 * Omega)
                )
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
                    jnp.cos(Omega)
                    * jnp.sin(prec)
                    * jnp.sin(obliq)
                    * jnp.sin(omega)
                    + (
                        -(jnp.cos(obliq) * jnp.sin(i))
                        + jnp.cos(i)
                        * jnp.cos(omega)
                        * jnp.sin(prec)
                        * jnp.sin(obliq)
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
            (jnp.cos(prec + omega) * jnp.sin(i) * jnp.sin(f - prec))
            / (-1 + f2) ** 2
            + (
                jnp.cos(f - prec)
                * (
                    -(
                        (-2 + f1)
                        * f1
                        * jnp.cos(i)
                        * jnp.cos(obliq)
                        * jnp.sin(obliq)
                    )
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
        a (Array [Rstar]): Semi-major axis of the orbit in units of R_star
        e (Array [Dimensionless]): Eccentricity of the orbit
        f (Array [Radian]): True anomaly, the angle between the direction of periapsis
                            and the current position of the planet as seen from
                            the star.
        Omega (Array [Radian]): Longitude of the ascending node
        i (Array [Radian]): Orbital inclination
        omega (Array [Radian]): Argument of periapsis
        r (Array [Rstar]): Equitorial radius of the planet, in units of R_star
        obliq (Array [Radian]): Obliquity, the angle between the planet's orbital plane
                                and its equatorial plane. Defined when the planet is at
                                periapsis with an Omega of zero as a rotation around the
                                sky-frame y-axis, such that a positive obliquity tips
                                the planet's north pole away from the star.
        prec (Array [Radian]): Precession angle, or equivalently the longitude of
                               ascending node of the planet's equatorial plane. This
                               is defined at periapsis with an Omega of zero as a
                               rotation about the sky-frame z-axis.
        f1 (Array [Dimensionless]): The flattening coefficient of the planet that
                                    describes the compression along the planet's polar
                                    axis. A value of 0.0 indicates no flattening. f1
                                    must always be larger than f2.
        f2 (Array [Dimensionless]): The flattening coefficient of the planet that
                                    describes the compression along the planet's
                                    "y" axis, the vector in its equatorial plane that is
                                    perpendicular to the direction of motion at
                                    periapsis, assuming the 0.0 obliquity. f2 must
                                    always be smaller than f1.
                                    
        **kwargs: Unused additional keyword arguments. These are included so that we
                    can can take in a larger state dictionary that includes all of the
                    required parameters along with other unnecessary ones.
    Returns:
        dict: A dictionary with keys representing different coefficient names and
              their corresponding values.
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
