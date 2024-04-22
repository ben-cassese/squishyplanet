import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from squishyplanet.engine.planet_3d import planet_3d_coeffs
from squishyplanet.engine.planet_2d import planet_2d_coeffs
from squishyplanet.engine.parametric_ellipse import poly_to_parametric
from squishyplanet.engine.greens_basis_transform import generate_change_of_basis_matrix
from squishyplanet.engine.kepler import kepler, skypos
from squishyplanet.engine.polynomial_limb_darkened_transit import lightcurve
from squishyplanet.engine.phase_curve_utils import (
    pre_squish_transform,
    generate_sample_radii_thetas,
    sample_surface,
    planet_surface_normal,
    lambertian_reflection,
    emission_profile,
    reflected_phase_curve,
    emission_phase_curve,
    phase_curve,
    stellar_ellipsoidal_variations,
    stellar_doppler_variations,
)

import copy
from functools import partial
import pprint


class OblateSystem:
    """
    The core user interface for ``squishyplanet``, used to model potentially-triaxial
    exoplanet transits/phase curves.

    Note, all instances will have values associated with phase curve
    calculations, such as albedo and hotspot location. However, if inputs such are
    "compute_reflected_phase_curve" are set to ``False``, these values will not be used.
    The :func:`lightcurve` method will return only the transit light curve in this case,
    and should be used if computing a transit, reflected, or emitted phase curve.

    All arguments will be internally converted to ``jax.numpy`` dtypes, and all methods
    will similarly return ``jax.numpy`` arrays. These can be treated similarly to
    numpy arrays in most cases, but if passing outputs to external inference libraries
    expecting numpy, you may need to explicitly convert them.

    Properties:
        state (dict):
            A dictionary of all the parameters of the system, including those specified
            by the user, default values, and those calculated by combinations of the
            two. Immutable, but can be accessed to see the current state of the system.

    Args:
        times (array-like, [Days], default=None):
            The times at which to calculate the light curve. The gap between times is
            assumed to be in units of days, but any zero-point/standard system
            (e.g. BJD) will work. A required parameter, will raise an error if not
            provided.
        t_peri (float, [Days], default=None):
            The time of periastron passage. A required parameter, will raise an error
            if not provided.
        period (float, [Days], default=None):
            The period of the orbit. A required parameter, will raise an error if not
            provided.
        a (float, [Rstar], default=None):
            The semi-major axis of the orbit in units of the radius of the star. A
            required parameter, will raise an error if not provided.
        e (float, default=0.0):
            The eccentricity of the orbit.
        i (float, [Radian], default=jnp.pi / 2):
            The inclination of the orbit.
        Omega (float, [Radian], default=jnp.pi):
            The longitude of the ascending node. Changing this will **not affect** the
            transit light curve (more accurately, changes can always be compensated for
            via rotations in obliq or prec). It is included only because it naturally
            arises in the orbit rotations, and I guess could come into play if anyone
            ever wants to do a joint astrometry model.
        omega (float, [Radian], default=0.0):
            The argument of periapsis. Set to 0.0 for a circular orbit, otherwise there
            will be degeneracies with t_peri.
        obliq (float, [Radian], default=0.0):
            The obliquity of the planet. This is the angle between the planet's rotation
            axis and the normal to the orbital plane. It is defined such that a planet
            on a circular orbit with :math:`\Omega = 0` and :math:`\\nu = 0` (i.e., when
            it's along the positive :math:`x` axis) will have its north pole tipped
            *away* from the star.
        prec (float, [Radian], default=0.0):
            The "precession angle" of the planet. This defined as a rotation of the
            planet about an axis that's aligned with its orbit normal and runs through
            the center of the planet (e.g., if obliq=0, it would set the planet's
            instantaneous rotational phase, and if obliq :math:`\\neq` 0, it would set
            the "season" of the northern hemisphere at periastron passage.)
        r (float, [Rstar], default=None):
            The equatorial radius of the planet. This will always be the largest of the
            3 axes of the triaxial ellipsoid. A required parameter, will raise an error
            if not provided.
        f1 (float, [Dimensionless], default=0.0):
            The fractional difference between the (longest) equatorial and polar radii
            of the planet. This is defined as :math:`(R_{eq} - R_{pol}) / R_{eq}`.
        f2 (float, [Dimensionless], default=0.0):
            The fractional difference between long and short radii of the ellipse that
            defines the equator of the planet. Defined similarly to f1.
        ld_u_coeffs (array-like, default=jnp.array([0.0, 0.0])):
            The coefficients that determine the limb darkening profile of the star. The
            star is assumed to be azimuthally symmetric and have a radial profile
            described by:

            .. math::

                \\frac{I(\mu)}{I_0} = - \Sigma_{i=0}^N u_i (1 - \mu)^i

            for some order polynomial :math:`N`. See
            `Agol, Luger, and Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_
            for more.
        hotspot_latitude (float, [Radian], default=0.0):
            The latitude of a potential hotspot on the planet. This is defined according
            to the "physics" convention of spherical coordinates, not in the geography
            sense: 0 is the north pole, :math:`\pi/2` is the equator, and :math:`\pi` is
            the south pole.
        hotspot_longitude (float, [Radian], default=0.0):
            The longitude of a potential hotspot on the planet.
        hotspot_concentration (float, default=0.2):
            The "concentration" of the hotspot. This is the :math:`\kappa` parameter in
            the von Mises-Fisher distribution that describes the hotspot.
        reflected_albedo (float, default=1.0):
            The (spatialy uniform) albedo of the planet. This is the fraction of light
            that is reflected, though the directional-dependent scattering is dictated
            by Lambert's cosine law.
        emitted_scale (float, default=1.0):
            The total emitted flux of the planet, in units of un-occulted stellar flux.
            The von Mises-Fisher distribution integrates to 1, and this factor scales
            the resulting emission profile.
        systematic_trend_coeffs (array-like, default=jnp.array([0.0,0.0])):
            The coefficients that determine the polynomial trend in time added to the
            lightcurves. Used to optionally model long-term drifts in observed data.
        log_jitter (float, default=-10):
            The log of the "jitter" term included in likelihood calculations. The jitter
            is added in quadrature to the provided uncertainties to account for any
            unmodeled noise in the data.
        tidally_locked (bool, default=True):
            Whether the planet is tidally locked to the star. If ``True``, then ``prec``
            will always be set equal to the true anomaly, meaning the same face of the
            planet will always face the star.
        compute_reflected_phase_curve (bool, default=False):
            Whether to include flux reflected by the planet when calling
            :func:`lightcurve`.
        compute_emitted_phase_curve (bool, default=False):
            Whether to include flux emitted by the planet when calling
            :func:`lightcurve`.
        compute_stellar_ellipsoidal_variations (bool, default=False):
            Whether to include stellar ellipsoidal variations in the light curve. This
            is the effect of the star's shape changing due to the gravitational pull of
            the planet, and here is modeled as a simple sinusoidal variation with 4
            peaks per orbit.
        compute_stellar_doppler_variations (bool, default=False):
            Whether to include stellar doppler variations in the light curve. This 
            captures the effects of the star's radial velocity changing and boosting the
            total flux/pushing some flux into/out of the bandpass of the observation.
            Here, it is modeled as a simple sinusoidal variation with 2 peaks per orbit.
        phase_curve_nsamples (int, default=50_000):
            The number of random samples of the planet's surface to draw when performing
            Monte Carlo estimates of the emitted/reflected flux. A larger number will
            increase the resolution/shrink the error of the estimate but result in
            longer computation times.
        random_seed (int, default=0):
            A random seed used for the Monte Carlo integrals in the phase curve. This
            feeds into ``jax.random.PRNGKey``. Runs with the same ``random_seed`` will
            always return identical outputs, so if checking the affect of altering
            ``phase_curve_nsamples``, you should change this as well.
        data (array-like, default=jnp.array([1.0])):
            The observed data to compare to the light curve. Must be the same length as
            ``times``. Only needed if calling :func:`loglike`.
        uncertainties (array-like, default=jnp.array([0.01])):
            The uncertainties on the observed data. Must be the same length as ``data``,
            even if the errors are homoskedastic. Only needed if calling
            :func:`loglike`.

    """

    def __init__(
        self,
        times=None,
        t_peri=None,
        period=None,
        a=None,
        e=0.0,
        i=jnp.pi / 2,
        Omega=jnp.pi,
        omega=0.0,
        obliq=0.0,
        prec=0.0,
        r=None,
        f1=0.0,
        f2=0.0,
        ld_u_coeffs=jnp.array([0.0, 0.0]),
        hotspot_latitude=0.0,
        hotspot_longitude=0.0,
        hotspot_concentration=0.2,
        reflected_albedo=1.0,
        emitted_scale=1e-6,
        stellar_ellipsoidal_alpha=1e-6,
        stellar_doppler_alpha=1e-6,
        systematic_trend_coeffs=jnp.array([0.0, 0.0]),
        log_jitter=-10,
        tidally_locked=True,
        compute_reflected_phase_curve=False,
        compute_emitted_phase_curve=False,
        compute_stellar_ellipsoidal_variations=False,
        compute_stellar_doppler_variations=False,
        phase_curve_nsamples=50_000,
        random_seed=0,
        data=jnp.array([1.0]),
        uncertainties=jnp.array([0.01]),
    ):

        state_keys = [
            "times",
            "t_peri",
            "period",
            "a",
            "e",
            "i",
            "Omega",
            "omega",
            "obliq",
            "prec",
            "r",
            "f1",
            "f2",
            "ld_u_coeffs",
            "hotspot_latitude",
            "hotspot_longitude",
            "hotspot_concentration",
            "reflected_albedo",
            "emitted_scale",
            "stellar_ellipsoidal_alpha",
            "stellar_doppler_alpha",
            "systematic_trend_coeffs",
            "log_jitter",
            "tidally_locked",
            "compute_reflected_phase_curve",
            "compute_emitted_phase_curve",
            "compute_stellar_ellipsoidal_variations",
            "compute_stellar_doppler_variations",
            "phase_curve_nsamples",
            "random_seed",
            "data",
            "uncertainties",
        ]

        state = {}
        for key in state_keys:
            state[key] = locals()[key]
        self._state = state

        self._validate_inputs()

        # necessary for all light curves
        self._state["greens_basis_transform"] = generate_change_of_basis_matrix(
            len(self._state["ld_u_coeffs"])
        )

        # everything below here is just an instantaneous snapshot mostly for plotting,
        # these will all vary with different parameter inputs
        time_deltas = self._state["times"] - self._state["t_peri"]
        mean_anomalies = 2 * jnp.pi * time_deltas / state["period"]
        true_anomalies = kepler(mean_anomalies, state["e"])
        self._state["f"] = true_anomalies

        if self._state["tidally_locked"]:
            self._state["prec"] = self._state["f"]

        positions = skypos(**state)
        state["x_c"] = positions[0, :]
        state["y_c"] = positions[1, :]
        state["z_c"] = positions[2, :]

        self._coeffs_3d = planet_3d_coeffs(**self._state)
        for key in self._coeffs_3d.keys():
            if self._coeffs_3d[key].shape[0] == ():
                self._coeffs_3d[key] = jnp.array([self._coeffs_3d[key]])

        self._coeffs_2d = planet_2d_coeffs(**self._coeffs_3d)
        self._para_coeffs_2d = poly_to_parametric(**self._coeffs_2d)

    def __repr__(self):
        s = pprint.pformat(self.state)
        return f"OblateSystem(\n{s}\n)"

    @property
    def state(self):
        """
        A dictionary that includes all of the parameters of the system.

        This is an immutable property, and will raise an error if you try to set it.
        If altering parameters that would affect a lightcurve, pass those as a
        dictionary to the :func:`lightcurve` method. If altering the data or times at
        which to generate the lightcurve, just define a new system with those values.

        Returns:
            dict:
            A dictionary of all the parameters of the system, including those specified
            by the user, default values, and those calculated by combinations of the
            two.
        """
        return self._state

    def _validate_inputs(self):
        for key, val in self._state.items():
            if type(val) == type(None):
                raise ValueError(f"'{key}' is a required parameter")

        self._state["ld_u_coeffs"] = jnp.array(self._state["ld_u_coeffs"])
        assert (
            self._state["ld_u_coeffs"].shape[0] >= 2
        ), "ld_u_coeffs must have at least 2 (even if higher-order terms are 0)"
        assert (
            type(self._state["phase_curve_nsamples"]) == int
        ), "phase_curve_nsamples must be an integer"
        assert type(self._state["random_seed"]) == int, "random_seed must be an integer"

        if self._state["e"] == 0:
            assert self._state["omega"] == 0, "omega must be 0 for a circular orbit"

        shapes = []
        for key in self._state.keys():
            if (
                (key == "times")
                | (key == "ld_u_coeffs")
                | (key == "phase_curve_nsamples")
                | (key == "random_seed")
                | (key == "data")
                | (key == "uncertainties")
                | (key == "systematic_trend_coeffs")
            ):
                continue
            elif type(self._state[key]) == bool:
                continue

            if (type(self._state[key]) == float) | (type(self._state[key]) == int):
                self._state[key] = jnp.array([self._state[key]])
                shapes.append(1)
            else:
                if len(self._state[key].shape) > 1:
                    raise ValueError(
                        "All parameters must be scalars or 1D arrays of the same shape."
                    )
                if self._state[key].shape == ():
                    self._state[key] = jnp.array([self._state[key]])
                    shapes.append(1)
                else:
                    shapes.append(self._state[key].shape[0])
        if len(jnp.unique(jnp.array(shapes))) > 2:
            raise ValueError(
                "All parameters must be scalars or arrays of the same shape."
            )

    def _illustrate_helper(self, times=None, true_anomalies=None, nsamples=50_000):

        if (times is not None) & (true_anomalies is not None):
            raise ValueError("Provide either times or true anomalies but not both")

        if times is not None:
            time_deltas = times - self._state["t_peri"]
            mean_anomalies = 2 * jnp.pi * time_deltas / self._state["period"]
            true_anomalies = kepler(mean_anomalies, self._state["e"])
        elif true_anomalies is not None:
            pass
        else:
            # true_anomalies = jnp.array(
            #     [0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]
            # )
            true_anomalies = jnp.array([jnp.pi / 2])

        if (type(true_anomalies) == float) | (type(true_anomalies) == int):
            true_anomalies = jnp.array([true_anomalies])

        # the trace of the orbit
        fs = jnp.linspace(0, 2 * jnp.pi, 200)
        positions = skypos(
            a=self._state["a"],
            e=self._state["e"],
            f=fs,
            Omega=self._state["Omega"],
            i=self._state["i"],
            omega=self._state["omega"],
        )

        original_state = copy.deepcopy(self._state)
        X_outline = []
        Y_outline = []
        Xs = []
        Ys = []
        Reflection = []
        Emission = []
        for i in range(len(true_anomalies)):

            # all of these could just be done in one go,
            # but bookkeeping was easier this way
            self._state["f"] = jnp.array([true_anomalies[i]])
            self._coeffs_3d = planet_3d_coeffs(**self._state)
            self._coeffs_2d = planet_2d_coeffs(**self._coeffs_3d)
            self._para_coeffs_2d = poly_to_parametric(**self._coeffs_2d)

            # the boundary of the planet
            thetas = jnp.linspace(0, 2 * jnp.pi, 200)
            x_outline = (
                self._para_coeffs_2d["c_x1"] * jnp.cos(thetas)
                + self._para_coeffs_2d["c_x2"] * jnp.sin(thetas)
                + self._para_coeffs_2d["c_x3"]
            )
            y_outline = (
                self._para_coeffs_2d["c_y1"] * jnp.cos(thetas)
                + self._para_coeffs_2d["c_y2"] * jnp.sin(thetas)
                + self._para_coeffs_2d["c_y3"]
            )

            # the phase curve bits
            sample_radii, sample_thetas = generate_sample_radii_thetas(
                jax.random.key(0), jnp.arange(nsamples)
            )
            x, y, z = sample_surface(
                sample_radii,
                sample_thetas,
                **self._coeffs_2d,
                **self._coeffs_3d,
            )

            # the reflected brightness profile
            normals = planet_surface_normal(x, y, z, **self._coeffs_3d)
            star_cos_ang = surface_star_cos_angle(
                normals,
                self._state["x_c"],
                self._state["y_c"],
                self._state["z_c"],
            )
            reflection = lambertian_reflection(star_cos_ang, x, y, z)

            # the emitted brightness profile
            # need to take the first index since you aren't scanning here
            transform = pre_squish_transform(**self._state)[0]
            emission = emission_profile(x, y, z, transform, **self._state)

            X_outline.append(x_outline)
            Y_outline.append(y_outline)
            Xs.append(x)
            Ys.append(y)
            Reflection.append(reflection)
            Emission.append(emission)

        X_outline = jnp.array(X_outline)
        Y_outline = jnp.array(Y_outline)
        Xs = jnp.array(Xs)
        Ys = jnp.array(Ys)
        Reflection = jnp.array(Reflection)
        Emission = jnp.array(Emission)

        self._state = original_state
        return {
            "orbit_positions": positions,
            "planet_x_outlines": X_outline,
            "planet_y_outlines": Y_outline,
            "sample_xs": Xs,
            "sample_ys": Ys,
            "reflected_intensity": Reflection,
            "emitted_intensity": Emission,
        }

    def lightcurve(self, params={}):
        """
        Compute the light curve of the system.

        This method will return the light curve of the system at the times specified
        when the system was initialized. If you want to compute the light curve at
        different times, or with different orbital parameters, you can pass those
        parameters as a dictionary to this method.

        The first time this is run for a given system, JAX will jit-compile the
        function, which can take some time. Subsequent calls will be much faster unless
        you change the shape of any of the input arrays (e.g., changing the number of
        times or the order of the polynomial limb darkening law). In those cases, or if
        changing any of boolean flags, JAX will need to re-compile the function again.


        Args:
            params (dict, default={}):
                A dictionary of parameters to update in the system state. Any keys
                not provided will be pulled from the current state of the system.

        Returns:
            Array: The timeseries lightcurve of the system. The length will be equal to
            `state["times"]`, and each index corresponds to a time in that array.

        Examples:
            >>> state = {
                    "t_peri" : 0.0,
                    "times" : jnp.linspace(-jnp.pi, 2*jnp.pi, 3504),
                    "a" : 2.0,
                    "period" : 2*jnp.pi,
                    "r" : 0.1,
                    "compute_reflected_phase_curve" : True,
                    "compute_emitted_phase_curve" : True,
                    "emitted_scale" : 1e-5,
                }
            >>> system = OblateSystem(**state)
            >>> system.lightcurve()
        """
        return _lightcurve(
            compute_reflected_phase_curve=self._state["compute_reflected_phase_curve"],
            compute_emitted_phase_curve=self._state["compute_emitted_phase_curve"],
            compute_stellar_ellipsoidal_variations=self._state[
                "compute_stellar_ellipsoidal_variations"
            ],
            compute_stellar_doppler_variations=self._state[
                "compute_stellar_doppler_variations"
            ],
            random_seed=self._state["random_seed"],
            phase_curve_nsamples=self._state["phase_curve_nsamples"],
            state=self._state,
            params=params,
        )

    def loglike(self, params={}):
        """
        Compute the log likelihood of the system given the observed data and some set of
        parameters.

        This method will call :func:`lightcurve` with the provided parameters and
        compare the output to the observed data. The likelihood is assumed to be
        Gaussian with no correlation between times. The jitter term is added in
        quadrature to the provided uncertainties.

        Args:
            params (dict, default={}):
                A dictionary of parameters to update in the system state. Any keys
                not provided will be pulled from the current state of the system.

        Returns:
            float:
            The log likelihood of the system given the observed data and the
            provided parameters.

        """
        return _loglike(
            compute_reflected_phase_curve=self._state["compute_reflected_phase_curve"],
            compute_emitted_phase_curve=self._state["compute_emitted_phase_curve"],
            compute_stellar_ellipsoidal_variations=self._state[
                "compute_stellar_ellipsoidal_variations"
            ],
            compute_stellar_doppler_variations=self._state[
                "compute_stellar_doppler_variations"
            ],
            random_seed=self._state["random_seed"],
            phase_curve_nsamples=self._state["phase_curve_nsamples"],
            state=self._state,
            params=params,
        )


@partial(
    jax.jit,
    static_argnums=(
        0,
        1,
        2,
        3,
        4,
        5,
    ),
)
def _lightcurve(
    compute_reflected_phase_curve,
    compute_emitted_phase_curve,
    compute_stellar_ellipsoidal_variations,
    compute_stellar_doppler_variations,
    random_seed,
    phase_curve_nsamples,
    state,
    params,
):
    # always compute the primary transit and trend
    for key in params.keys():
            state[key] = params[key]
    transit = lightcurve(state)
    trend = jnp.polyval(state["systematic_trend_coeffs"], state["times"])

    # if you don't want any phase curve stuff, you're done
    if (not compute_reflected_phase_curve) & (not compute_emitted_phase_curve) and \
        (not compute_stellar_doppler_variations) & (not compute_stellar_ellipsoidal_variations):
        return transit + trend

    ######################################################
    # compute the planet's contribution to the phase curve
    ######################################################

    # generate the radii and thetas that you'll reuse at each timestep
    sample_radii, sample_thetas = generate_sample_radii_thetas(
        jax.random.key(random_seed),
        jnp.arange(phase_curve_nsamples),
    )
    
    # technically these are all calculated in "transit", but since phase
    # curves are a) rare and b) expensive, we'll just do it again to keep
    # the transit section of the code more self-contained
    three = planet_3d_coeffs(**state)
    two = planet_2d_coeffs(**three)
    positions = skypos(**state)
    x_c = positions[0, :]
    y_c = positions[1, :]
    z_c = positions[2, :]

    # just the reflected component
    if compute_reflected_phase_curve & (not compute_emitted_phase_curve):
        reflected = reflected_phase_curve(
            sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
        )
        emitted = 0.0
        # it really didn't make a difference in speed here
        # reflected = jax.vmap(
        #     reflected_phase_curve, in_axes=(0, 0, None, None, None, None, None, None)
        # )(sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c)
        # reflected = jnp.mean(reflected, axis=0)

    # just the emitted component
    elif (not compute_reflected_phase_curve) & compute_emitted_phase_curve:
        reflected = 0.0
        emitted = emission_phase_curve(sample_radii, sample_thetas, two, three, state)

    # both reflected and emitted components. this function shares some of the
    # computation between the two, so it's a bit faster than running them separately
    elif compute_reflected_phase_curve & compute_emitted_phase_curve:
        reflected, emitted = phase_curve(
            sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
        )

    ####################################################
    # compute the star's contribution to the phase curve
    ####################################################

    if compute_stellar_ellipsoidal_variations | compute_stellar_doppler_variations:
        time_deltas = state["times"] - state["t_peri"]
        mean_anomalies = 2 * jnp.pi * time_deltas / state["period"]
        true_anomalies = kepler(mean_anomalies, state["e"])

    if compute_stellar_ellipsoidal_variations:
        ellipsoidal = stellar_ellipsoidal_variations(true_anomalies, state["stellar_ellipsoidal_alpha"], state["period"])
    else:
        ellipsoidal = 0.0
    
    if compute_stellar_doppler_variations:
        doppler = stellar_doppler_variations(true_anomalies, state["stellar_doppler_alpha"], state["period"])
    else:
        doppler = 0.0

    # put it all together
    return transit + trend + reflected + emitted + ellipsoidal + doppler


@partial(
    jax.jit,
    static_argnums=(
        0,
        1,
        2,
        3,
        4,
        5,
    ),
)
def _loglike(
    compute_reflected_phase_curve,
    compute_emitted_phase_curve,
    compute_stellar_ellipsoidal_variations,
    compute_stellar_doppler_variations,
    random_seed,
    phase_curve_nsamples,
    state,
    params,
):
    lc = _lightcurve(
        compute_reflected_phase_curve,
        compute_emitted_phase_curve,
        random_seed,
        phase_curve_nsamples,
        state,
        params,
    )

    for key in params.keys():
        state[key] = params[key]

    resids = state["data"] - lc
    var = jnp.exp(state["log_jitter"]) + state["uncertainties"] ** 2

    return jnp.sum(-0.5 * (resids**2 / var + jnp.log(var)))
