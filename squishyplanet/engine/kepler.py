# this is a fork of jaxoplanet/src/jaxoplanet/core/kepler.py, many thanks to the original authors

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.interpreters import ad


@jax.jit
def kepler(M, ecc):
    """Solve Kepler's equation to compute the true anomaly.

    This implementation is based on that within `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet/>`_, many thanks to the authors.

    Args:
        M (Array [Radian]): Mean anomaly
        ecc (Array [Dimensionless]): Eccentricity

    Returns:
        Array: True anomaly in radians
    """
    sinf, cosf = _kepler(M, ecc)
    # this is the only bit that's different from jaxoplanet-
    # puts true anomalies into the range [0, 2*pi)
    f = jnp.arctan2(sinf, cosf)
    return jnp.where(f < 0, f + 2 * jnp.pi, f)


@jax.custom_jvp
def _kepler(M, ecc):
    # Wrap into the right range
    M = M % (2 * jnp.pi)

    # We can restrict to the range [0, pi)
    high = M > jnp.pi
    M = jnp.where(high, 2 * jnp.pi - M, M)

    # Solve
    ome = 1 - ecc
    E = _starter(M, ecc, ome)
    E = _refine(M, ecc, ome, E)

    # Re-wrap back into the full range
    E = jnp.where(high, 2 * jnp.pi - E, E)

    # Convert to true anomaly; tan(0.5 * f)
    tan_half_f = jnp.sqrt((1 + ecc) / (1 - ecc)) * jnp.tan(0.5 * E)
    tan2_half_f = jnp.square(tan_half_f)

    # Then we compute sin(f) and cos(f) using:
    #  sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    #  cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tan2_half_f)
    sinf = 2 * tan_half_f * denom
    cosf = (1 - tan2_half_f) * denom

    return sinf, cosf


@_kepler.defjvp
def _(primals, tangents):
    M, e = primals
    M_dot, e_dot = tangents
    sinf, cosf = _kepler(M, e)

    # Pre-compute some things
    ecosf = e * cosf
    ome2 = 1 - e**2

    def make_zero(tan):
        if type(tan) is ad.Zero:
            return ad.zeros_like_aval(tan.aval)
        else:
            return tan

    # Propagate the derivatives
    f_dot = make_zero(M_dot) * (1 + ecosf) ** 2 / ome2**1.5
    f_dot += make_zero(e_dot) * (2 + ecosf) * sinf / ome2

    return (sinf, cosf), (cosf * f_dot, -sinf * f_dot)


def _starter(M, ecc, ome):
    M2 = jnp.square(M)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - M) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * M
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = jnp.square(jnp.cbrt(jnp.abs(r) + jnp.sqrt(q2 * q + r * r)))
    return (2 * r * w / (jnp.square(w) + w * q + q2) + M) / d


def _refine(M, ecc, ome, E):
    sE = E - jnp.sin(E)
    cE = 1 - jnp.cos(E)

    f_0 = ecc * sE + E * ome - M
    f_1 = ecc * cE + ome
    f_2 = ecc * (E - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24)

    return E + dE


def _x(a, e, f, Omega, i, omega):
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


def _y(a, e, f, Omega, i, omega):
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


def _z(a, e, f, Omega, i, omega):
    return -((a * (-1 + e**2) * jnp.sin(i) * jnp.sin(f + omega)) / (1 + e * jnp.cos(f)))


@jax.jit
def skypos(a, e, f, Omega, i, omega, **kwargs):
    """
    Compute the cartesian coordinates of the center of the planet in the sky frame
    given its orbital elements.

    Args:
        a (Array [Rstar]): Semi-major axis of the orbit in units of R_star
        e (Array [Dimensionless]): Eccentricity of the orbit
        f (Array [Radian]): True anomaly, the angle between the direction of periapsis
                            and the current position of the planet as seen from
                            the star.
        Omega (Array [Radian]): Longitude of the ascending node
        i (Array [Radian]): Orbital inclination
        omega (Array [Radian]): Argument of periapsis
        **kwargs: Unused additional keyword arguments. These are included so that we
            can can take in a larger state dictionary that includes all of the
            required parameters along with other unnecessary ones.

    Returns:
        Array: The cartesian coordinates of the planet in the sky frame. Shape [3, N].
    """
    return jnp.array(
        [
            _x(a, e, f, Omega, i, omega),
            _y(a, e, f, Omega, i, omega),
            _z(a, e, f, Omega, i, omega),
        ]
    )


def true_anomaly_at_transit_center(e, i, omega):
    """
    Computes the true anomaly at the instant of minimum star/planet separation.

    Uses equations 4.12-4.18 of
    `Kipping 2011 <https://ui.adsabs.harvard.edu/abs/2011PhDT.......294K/abstract>`_
    to compute the true anomaly at the instant of minimum star/planet separation.

    Args:
        e (Array [Dimensionless]): Eccentricity of the orbit
        i (Array [Radian]): Orbital inclination
        omega (Array [Radian]): Argument of periapsis

    Returns:
        Array: True anomaly at the instant of minimum star/planet separation in radians.
    """

    hp = e * jnp.sin(omega)
    kp = e * jnp.cos(omega)

    eta_1 = (kp / (1 + hp)) * (jnp.cos(i) ** 2)
    eta_2 = (kp / (1 + hp)) * (1 / (1 + hp)) * (jnp.cos(i) ** 2) ** 2
    eta_3 = (
        -(kp / (1 + hp))
        * ((-6 * (1 + hp) + kp**2 * (-1 + 2 * hp)) / (6 * (1 + hp) ** 3))
        * (jnp.cos(i) ** 2) ** 3
    )
    eta_4 = (
        -(kp / (1 + hp))
        * ((-2 * (1 + hp) + kp**2 * (-1 + 3 * hp)) / (2 * (1 + hp) ** 4))
        * (jnp.cos(i) ** 2) ** 4
    )
    eta_5 = (
        (kp / (1 + hp))
        * (
            (
                40 * (1 + hp) ** 2
                - 40 * kp**2 * (-1 + 3 * hp + 4 * hp**2)
                + kp**4 * (3 - 19 * hp + 8 * hp**2)
            )
            / (40 * (1 + hp) ** 6)
        )
        * (jnp.cos(i) ** 2) ** 5
    )
    eta_6 = (
        (kp / (1 + hp))
        * (
            (
                24 * (1 + hp) ** 2
                - 40 * kp**2 * (-1 + 4 * hp + 5 * hp**2)
                + 9 * kp**4 * (1 - 8 * hp + 5 * hp**2)
            )
            / (24 * (1 + hp) ** 7)
        )
        * (jnp.cos(i) ** 2) ** 6
    )

    return jnp.pi / 2 - omega - eta_1 - eta_2 - eta_3 - eta_4 - eta_5 - eta_6


def t0_to_t_peri(e, i, omega, period, t0, **kwargs):
    f = true_anomaly_at_transit_center(e, i, omega)

    eccentric_anomaly = jnp.arctan2(jnp.sqrt(1 - e**2) * jnp.sin(f), e + jnp.cos(f))
    mean_anomaly = eccentric_anomaly - e * jnp.sin(eccentric_anomaly)

    return t0 - period / (2 * jnp.pi) * mean_anomaly
