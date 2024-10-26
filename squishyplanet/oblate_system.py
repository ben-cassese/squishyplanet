import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import copy
from functools import partial
import pprint
import matplotlib.pyplot as plt

from squishyplanet.engine.planet_3d import (
    planet_3d_coeffs,
    extended_illumination_offsets,
    planet_3d_coeffs_extended_illumination,
)
from squishyplanet.engine.planet_2d import planet_2d_coeffs
from squishyplanet.engine.parametric_ellipse import (
    poly_to_parametric_helper,
    poly_to_parametric,
)
from squishyplanet.engine.greens_basis_transform import generate_change_of_basis_matrix
from squishyplanet.engine.kepler import kepler, skypos, t0_to_t_peri
from squishyplanet.engine.polynomial_limb_darkened_transit import (
    lightcurve as poly_lightcurve,
)
from squishyplanet.engine.polynomial_limb_darkened_transit import parameterize_2d_helper

from squishyplanet.engine.phase_curve_utils import (
    pre_squish_transform,
    generate_sample_radii_thetas,
    sample_surface,
    planet_surface_normal,
    surface_star_cos_angle,
    lambertian_reflection,
    corrected_emission_profile,
    reflected_phase_curve,
    extended_illumination_reflected_phase_curve,
    emission_phase_curve,
    phase_curve,
    stellar_ellipsoidal_variations,
    stellar_doppler_variations,
)


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
            The time of periastron passage. One of ``t_peri`` or ``t0`` must be
            provided.
        t0 (float, [Days], default=None):
            The time of transit center. One of ``t_peri`` or ``t0`` must be provided.
        period (float, [Days], default=None):
            The period of the orbit. A required parameter, will raise an error if not
            provided.
        a (float, [Rstar], default=None):
            The semi-major axis of the orbit in units of the radius of the star. A
            required parameter, will raise an error if not provided.
        tidally_locked (bool, default=None):
            Whether the planet is tidally locked to the star. If ``True``, then ``prec``
            will always be set equal to the true anomaly, meaning the same face of the
            planet will always face the star. A required parameter, will raise an error
            if not provided.
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
            on a circular orbit with :math:`\\Omega = 0` and :math:`\\nu = 0` (i.e., when
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
            3 axes of the triaxial ellipsoid. Either this or the entire set of
            ``projected_effective_r``, ``projected_f``, and ``projected_theta`` must be
            provided.
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

                \\frac{I(\\mu)}{I_0} = - \\Sigma_{i=0}^N u_i (1 - \\mu)^i

            for some order polynomial :math:`N`. See
            `Agol, Luger, and Foreman-Mackey 2020
            <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_ for more.
        hotspot_latitude (float, [Radian], default=0.0):
            The latitude of a potential hotspot on the planet. This is defined according
            to the "physics" convention of spherical coordinates, not in the geography
            sense: 0 is the north pole, :math:`\\pi/2` is the equator, and :math:`\\pi` is
            the south pole.
        hotspot_longitude (float, [Radian], default=0.0):
            The longitude of a potential hotspot on the planet.
        hotspot_concentration (float, default=0.2):
            The "concentration" of the hotspot. This is the :math:`\\kappa` parameter in
            the von Mises-Fisher distribution that describes the hotspot.
        albedo (float, default=1.0):
            The (spatialy uniform) albedo of the planet. This is the fraction of light
            that is reflected, though the directional-dependent scattering is dictated
            by Lambert's cosine law.
        emitted_scale (float, default=1e-5):
            A scale factor that sets the amplitude of the the emitted flux of the
            planet.
        systematic_trend_coeffs (array-like, default=jnp.array([0.0,0.0])):
            The coefficients that determine the polynomial trend in time added to the
            lightcurves. Used to optionally model long-term drifts in observed data.
        log_jitter (float, default=-jnp.inf):
            The log of the "jitter" term included in likelihood calculations. The jitter
            is added in quadrature to the provided uncertainties to account for any
            unmodeled noise in the data. This value is the *standard deviation* of the
            jitter, not the variance. If set to -jnp.inf, the jitter term will not
            affect the likelihood.
        projected_effective_r (float, [Rstar], default=0.0):
            The radius of a circle with the same area is the projected ellipse. This is
            only relevant if ``parameterize_with_projected_ellipse`` is set to ``True``,
            which will override ``r``, ``f1``, ``f2``, ``obliq``, and ``prec``.
        projected_f (float, [Dimensionless], default=0.0):
            The flattening value of the projected ellipse. This is only relevant if
            ``parameterize_with_projected_ellipse`` is set to ``True``, which will
            override ``r``, ``f1``, ``f2``, ``obliq``, and ``prec``.
        projected_theta (float, [Radian], default=0.0):
            The angle of the semi-major axis of the projected ellipse. This is only
            relevant if ``parameterize_with_projected_ellipse`` is set to ``True``,
            which will override ``r``, ``f1``, ``f2``, ``obliq``, and ``prec``.
        extended_illumination_npts (int, default=1):
            The number of points used to sample the star's projected disk as seen by
            the planet when calculating the reflected flux and accounting for the star's
            extended size. Closely follows the implementation in ``starry``,
            specifically Sec. 4.1 of `Luger et al. 2022
            <https://ui.adsabs.harvard.edu/abs/2022AJ....164....4L/abstract>`_.
            IMPLEMENTATION IS INCOMPLETE, WILL RAISE A NOTIMPLEMENTEDERROR IF SET TO
            ANYTHING OTHER THAN 1.
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
        parameterize_with_projected_ellipse (bool, default=False):
            Whether to parameterize the planet as a projected ellipse rather than a
            triaxial ellipsoid. If ``True``, then ``projected_effective_r``,
            ``projected_f``, and ``projected_theta`` will be used.
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
        exposure_time (float, [Days], default=0.0):
            The length of each exposure in the light curve, used to correct for finite
            integration times if ``oversample`` is set to a value greater than 1.
            Important: the finite exposure time correction procedure assumes that the
            given times correspond to the **midpoints** of each exposure, not the
            *start* or *end*. No checks are made to ensure that it is shorter than the
            minimum time difference between the provided times.
        oversample (int, default=1):
            The factor by which to oversample the light curve to partially compensate
            for finite-time integrations. The overdense lightcurve is then binned down
            to the original provided times. See e.g. `Kipping 2010
            <https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract>`_, "Binning
            is Sinning" for more. Must be a positive integer. Will be rounded up to
            nearest odd number.
        oversample_correction_order (int, default=2):
            After oversampling the light curve, how do you want to integrate over the
            exposure time to get the final binned light curve? This follows ``starry``'s
            treatment very closely: 0 is a centered Riemann sum like in Kipping 2010,
            1 is a trapezoidal rule, and 2 is Simpson's rule. Must be one of those
            values.

    """

    def __init__(
        self,
        times=None,
        t_peri=None,
        t0=None,
        period=None,
        a=None,
        tidally_locked=None,
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
        albedo=1.0,
        emitted_scale=1e-5,
        stellar_ellipsoidal_alpha=1e-6,
        stellar_doppler_alpha=1e-6,
        systematic_trend_coeffs=jnp.array([0.0, 0.0]),
        log_jitter=-jnp.inf,
        projected_effective_r=0.0,
        projected_f=0.0,
        projected_theta=0.0,
        extended_illumination_npts=1,
        compute_reflected_phase_curve=False,
        compute_emitted_phase_curve=False,
        compute_stellar_ellipsoidal_variations=False,
        compute_stellar_doppler_variations=False,
        parameterize_with_projected_ellipse=False,
        phase_curve_nsamples=50_000,
        random_seed=0,
        data=jnp.array([1.0]),
        uncertainties=jnp.array([jnp.inf]),
        exposure_time=0.0,
        oversample=1,
        oversample_correction_order=2,
    ):

        #######################################################################
        # setup
        #######################################################################

        state_keys = list(locals().keys())
        state_keys.remove("self")

        state = {}
        for key in state_keys:
            state[key] = locals()[key]
        self._state = state

        self._validate_inputs()

        #######################################################################
        # 1-time calculations
        #######################################################################

        # necessary for all light curves
        self._state["greens_basis_transform"] = generate_change_of_basis_matrix(
            len(self._state["ld_u_coeffs"])
        )

        # for oversampling
        if self._state["oversample"] > 1:

            self._state["oversample"] += 1 - self._state["oversample"] % 2
            self._state["stencil"] = jnp.ones(self._state["oversample"])

            # Construct the exposure time integration stencil
            if self._state["oversample_correction_order"] == 0:
                dt = jnp.linspace(-0.5, 0.5, 2 * self._state["oversample"] + 1)[1:-1:2]
            elif self._state["oversample_correction_order"] == 1:
                dt = jnp.linspace(-0.5, 0.5, self._state["oversample"])
                self._state["stencil"] = self._state["stencil"].at[1:-1].set(2)
            elif self._state["oversample_correction_order"] == 2:
                dt = jnp.linspace(-0.5, 0.5, self._state["oversample"])
                self._state["stencil"] = self._state["stencil"].at[1:-1:2].set(4)
                self._state["stencil"] = self._state["stencil"].at[2:-1:2].set(2)

            self._state["stencil"] = self._state["stencil"] / jnp.sum(
                self._state["stencil"]
            )

            dt = self._state["exposure_time"] * dt
            t = self._state["times"][:, None] + dt[None, :]
            t = t.reshape(-1)
            self._state["times"] = t

        else:
            self._state["times"] = self._state["times"]
            self._state["stencil"] = (
                None  # never used in this case, but to keep the state consistent
            )

        # # for extended illumination reflection curves
        # actually, still having trouble with this, so setting aside for now--
        # leaving it for a specific enhancement after 0.1.0
        if self._state["compute_reflected_phase_curve"] > 1:
            raise NotImplementedError(
                "Extended illumination reflection curves are not yet implemented"
            )
        # # based on starry._core.core.py's OpsReflected(OpsYlm) class
        # # create a grid of points uniformly distributed on the projected disk of the
        # # sta from an observer along the z-axis
        # N = int(2 + jnp.sqrt(self._state["extended_illumination_npts"] * 4 / jnp.pi))

        # # note these points will be squished closer together during calculations to
        # # account for the a-dependent extent of the star from the planet's perspective
        # dx = jnp.linspace(-1 + 1e-12, 1 - 1e-12, N)
        # dx, dy = jnp.meshgrid(dx, dx)
        # # dz = jnp.sqrt(1 - dx**2 - dy**2)
        # dz = 1 - dx**2 - dy**2
        # source_dx = dx[dz > 0].flatten()
        # source_dy = dy[dz > 0].flatten()
        # source_dz = dz[dz > 0].flatten()
        # pts = jnp.array([source_dx, source_dy, jnp.sqrt(source_dz)]).T
        # self._state["extended_illumination_points"] = pts
        # self._state["extended_illumination_npts"] = len(pts)

        # everything below here is just an instantaneous snapshot mostly for plotting,
        # these will all vary with different parameter inputs
        if self._state["t_peri"] is None:
            tp = t0_to_t_peri(**self._state)
        else:
            tp = self._state["t_peri"]
        time_deltas = self._state["times"] - tp
        mean_anomalies = 2 * jnp.pi * time_deltas / state["period"]
        true_anomalies = kepler(mean_anomalies, state["e"])
        self._state["f"] = true_anomalies

        if self._state["tidally_locked"]:
            self._state["prec"] = self._state["f"]

        positions = skypos(**state)
        self._state["x_c"] = positions[0, :]
        self._state["y_c"] = positions[1, :]
        self._state["z_c"] = positions[2, :]

        if not self._state["parameterize_with_projected_ellipse"]:
            self._coeffs_3d = planet_3d_coeffs(**self._state)
            for key in self._coeffs_3d.keys():
                if self._coeffs_3d[key].shape[0] == ():
                    self._coeffs_3d[key] = jnp.array([self._coeffs_3d[key]])

            self._coeffs_2d = planet_2d_coeffs(**self._coeffs_3d)
            self._para_coeffs_2d = poly_to_parametric(**self._coeffs_2d)

            r1, r2, _, _, cosa, sina = poly_to_parametric_helper(**self._coeffs_2d)
            area = jnp.pi * r1 * r2
            effective_r = jnp.sqrt(area / jnp.pi)
            self._state["projected_effective_r"] = effective_r
            effective_theta = jnp.arctan(sina / cosa)
            effective_theta = jnp.where(
                effective_theta < 0, effective_theta + jnp.pi, effective_theta
            )
            self._state["projected_theta"] = effective_theta
            effective_f = (
                jnp.max(jnp.array([r1, r2])) - jnp.min(jnp.array([r1, r2]))
            ) / jnp.max(jnp.array([r1, r2]))
            self._state["projected_f"] = effective_f
        else:
            self._coeffs_3d = {}
            area = jnp.pi * self._state["projected_effective_r"] ** 2
            r1 = jnp.sqrt(area / ((1 - self._state["projected_f"]) * jnp.pi))
            r2 = r1 * (1 - self._state["projected_f"])
            self._coeffs_2d, self._para_coeffs_2d = parameterize_2d_helper(
                projected_r=r1,
                projected_f=self._state["projected_f"],
                projected_theta=self._state["projected_theta"],
                xc=self._state["x_c"],
                yc=self._state["y_c"],
            )

        self._lightcurve_fwd_grad_enforced = self._setup_lightcurve_func()
        self._loglike_fwd_grad_enforced = self._setup_loglike_func()

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

        # we internally changed "times" if oversample > 1, but we alwasy bin it back
        # down to the original times, so we can undo that expansion here
        s = copy.deepcopy(self._state)
        if s["oversample"] > 1:
            s["times"] = (
                s["times"].reshape(-1, s["oversample"]) * s["stencil"][None, :]
            ).sum(axis=1)
        return s

    def _validate_inputs(self):
        for key, val in self._state.items():
            if type(val) == type(None):
                if key == "r":
                    if self._state["parameterize_with_projected_ellipse"]:
                        self._state["r"] = 0.0
                        continue
                if key == "t_peri":
                    continue
                if key == "t0":
                    continue
                raise ValueError(f"'{key}' is a required parameter")

        assert (self._state["t_peri"] is None) != (
            self._state["t0"] is None
        ), "Exactly one of 't_peri' or 't0' must be specified"

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
                | (key == "exposure_time")
                | (key == "oversample")
                | (key == "oversample_correction_order")
                | (key == "extended_illumination_npts")
            ):
                continue
            elif type(self._state[key]) == bool:
                continue

            if (type(self._state[key]) == float) | (type(self._state[key]) == int):
                self._state[key] = jnp.array([self._state[key]])
                shapes.append(1)
            else:
                if self._state[key] is None:
                    continue  # still one None hanging around, either t0 or t_peri
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

        if self._state["parameterize_with_projected_ellipse"]:
            assert self._state["projected_effective_r"] > 0, (
                "projected_effective_r must be greater than 0 if "
                "parameterize_with_projected_ellipse is True"
            )
            assert not (
                self._state["compute_reflected_phase_curve"]
                | self._state["compute_emitted_phase_curve"]
                | self._state["compute_stellar_ellipsoidal_variations"]
                | self._state["compute_stellar_doppler_variations"]
            ), (
                "parameterize_with_projected_ellipse is incompatible with phase "
                "curve calculations"
            )

            assert self._state["tidally_locked"] == False, (
                "parameterize_with_projected_ellipse is incompatible with "
                "tidally_locked=True"
            )

        assert (self._state["oversample_correction_order"] in [0, 1, 2]) & (
            type(self._state["oversample_correction_order"]) == int
        ), "oversample_correction_order must be 0, 1, or 2"

        assert self._state["oversample"] > 0, "oversample must be greater than 0"

        if self._state["oversample"] > 1:
            assert (
                self._state["exposure_time"] is not None
            ), "exposure_time must be provided if oversample > 1"

        if self._state["compute_stellar_ellipsoidal_variations"]:
            assert (
                self._state["e"] == 0.0
            ), "Stellar ellipsoidal variations are only valid for circular orbits"

        if self._state["compute_stellar_doppler_variations"]:
            assert (
                self._state["e"] == 0.0
            ), "Stellar doppler variations are only valid for circular orbits"

    def _setup_lightcurve_func(self):

        constants = {
            "tidally_locked": self._state["tidally_locked"],
            "compute_reflected_phase_curve": self._state[
                "compute_reflected_phase_curve"
            ],
            "compute_emitted_phase_curve": self._state["compute_emitted_phase_curve"],
            "compute_stellar_ellipsoidal_variations": self._state[
                "compute_stellar_ellipsoidal_variations"
            ],
            "compute_stellar_doppler_variations": self._state[
                "compute_stellar_doppler_variations"
            ],
            "parameterize_with_projected_ellipse": self._state[
                "parameterize_with_projected_ellipse"
            ],
            "oversample": self._state["oversample"],
            "random_seed": self._state["random_seed"],
            "phase_curve_nsamples": self._state["phase_curve_nsamples"],
            "extended_illumination_npts": self._state["extended_illumination_npts"],
            "state": self._state,
        }

        frozen = jax.tree_util.Partial(_lightcurve, **constants)

        @jax.custom_vjp
        def lightcurve(params):
            return frozen(params)

        def lightcurve_fwd(params):
            output = frozen(params)
            jac = jax.jacfwd(frozen)(params)
            return output, (jac,)

        def lightcurve_bwd(res, g):
            jac = res
            val = jax.tree.map(lambda x: x.T @ g, jac)
            return val

        lightcurve.defvjp(lightcurve_fwd, lightcurve_bwd)
        lightcurve = jax.jit(lightcurve)

        return lightcurve

    def _setup_loglike_func(self):

        constants = {
            "tidally_locked": self._state["tidally_locked"],
            "compute_reflected_phase_curve": self._state[
                "compute_reflected_phase_curve"
            ],
            "compute_emitted_phase_curve": self._state["compute_emitted_phase_curve"],
            "compute_stellar_ellipsoidal_variations": self._state[
                "compute_stellar_ellipsoidal_variations"
            ],
            "compute_stellar_doppler_variations": self._state[
                "compute_stellar_doppler_variations"
            ],
            "parameterize_with_projected_ellipse": self._state[
                "parameterize_with_projected_ellipse"
            ],
            "oversample": self._state["oversample"],
            "random_seed": self._state["random_seed"],
            "phase_curve_nsamples": self._state["phase_curve_nsamples"],
            "extended_illumination_npts": self._state["extended_illumination_npts"],
            "state": self._state,
        }

        frozen = jax.tree_util.Partial(_loglike, **constants)

        @jax.custom_vjp
        def loglike(params):
            return frozen(params)

        def loglike_fwd(params):
            output = frozen(params)
            jac = jax.jacfwd(frozen)(params)
            return output, jac

        def loglike_bwd(res, g):
            val = jax.tree.map(lambda x: x.T * g, res)
            return (val,)

        loglike.defvjp(loglike_fwd, loglike_bwd)
        loglike = jax.jit(loglike)

        return loglike

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
            true_anomalies = jnp.array([jnp.pi / 2])

        if (type(true_anomalies) == float) | (type(true_anomalies) == int):
            true_anomalies = jnp.array([true_anomalies])

        # the trace of the orbit
        fs = jnp.linspace(0, 2 * jnp.pi, 300)
        orbit_positions = skypos(
            a=self._state["a"],
            e=self._state["e"],
            f=fs,
            Omega=self._state["Omega"],
            i=self._state["i"],
            omega=self._state["omega"],
        )
        behind_star = (
            (orbit_positions[0, :] ** 2 + orbit_positions[1, :] ** 2) < 1
        ) & (orbit_positions[2, :] < 0)
        orbit_positions = orbit_positions.at[:, behind_star].set(jnp.nan)

        original_state = copy.deepcopy(self._state)
        original_3d_coeffs = copy.deepcopy(self._coeffs_3d)
        original_2d_coeffs = copy.deepcopy(self._coeffs_2d)
        original_para_coeffs_2d = copy.deepcopy(self._para_coeffs_2d)

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
            if self._state["tidally_locked"]:
                self._state["prec"] = self._state["f"]
            # self._coeffs_3d = planet_3d_coeffs(**self._state)
            # self._coeffs_2d = planet_2d_coeffs(**self._coeffs_3d)
            # self._para_coeffs_2d = poly_to_parametric(**self._coeffs_2d)
            positions = skypos(**self._state)
            self._state["x_c"] = positions[0, :]
            self._state["y_c"] = positions[1, :]
            self._state["z_c"] = positions[2, :]
            if not self._state["parameterize_with_projected_ellipse"]:
                self._coeffs_3d = planet_3d_coeffs(**self._state)
                for key in self._coeffs_3d.keys():
                    if self._coeffs_3d[key].shape[0] == ():
                        self._coeffs_3d[key] = jnp.array([self._coeffs_3d[key]])

                self._coeffs_2d = planet_2d_coeffs(**self._coeffs_3d)
                self._para_coeffs_2d = poly_to_parametric(**self._coeffs_2d)

            else:
                self._coeffs_3d = {}
                area = jnp.pi * self._state["projected_effective_r"] ** 2
                r1 = jnp.sqrt(area / ((1 - self._state["projected_f"]) * jnp.pi))
                r2 = r1 * (1 - self._state["projected_f"])
                self._coeffs_2d, self._para_coeffs_2d = parameterize_2d_helper(
                    projected_r=r1,
                    projected_f=self._state["projected_f"],
                    projected_theta=self._state["projected_theta"],
                    xc=self._state["x_c"],
                    yc=self._state["y_c"],
                )

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

            if not self._state["parameterize_with_projected_ellipse"]:
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
                emission = corrected_emission_profile(
                    x,
                    y,
                    z,
                    transform,
                    **self._state,
                )

                behind_star = ((x**2 + y**2) < 1) & (z < 0)
                reflection = jnp.where(behind_star, jnp.nan, reflection)
                emission = jnp.where(behind_star, jnp.nan, emission)

            else:
                x = jnp.nan
                y = jnp.nan
                reflection = jnp.nan
                emission = jnp.nan

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

        # behind_star = ((Xs ** 2 + Ys ** 2) < 1)
        # Reflection = jnp.where(Reflection == 0, jnp.nan, Reflection)
        # Emission = jnp.where(Emission == 0, jnp.nan, Emission)

        self._state = original_state
        self._coeffs_3d = original_3d_coeffs
        self._coeffs_2d = original_2d_coeffs
        self._para_coeffs_2d = original_para_coeffs_2d

        return {
            "orbit_positions": orbit_positions,
            "planet_x_outlines": X_outline,
            "planet_y_outlines": Y_outline,
            "sample_xs": Xs,
            "sample_ys": Ys,
            "reflected_intensity": Reflection,
            "emitted_intensity": Emission,
        }

    def illustrate(
        self,
        times=None,
        true_anomalies=None,
        orbit=True,
        reflected=False,
        emitted=False,
        star_fill=True,
        window_size=0.4,
        star_centered=False,
        nsamples=50_000,
        figsize=(8, 8),
    ):
        """
        Visualize the layout of the system at one or more times.

        This method, if run in a jupyter notebook, will display a plot of some
        combination of the star, planet, and its orbit. It can color in the planet
        according to its reflected or emission profile, and the star according to its
        limb darkening profile. Helpful for checking the orientation of planet hotspots
        and/or its orientation after deformation.

        Args:

            times (array-like, [Days], default=None):
                The times at which to illustrate the system. The gap between times is
                assumed to be in units of days, but any zero-point/standard system
                (e.g. BJD) will work. Provide either this or ``true_anomalies`` but not
                both.
            true_anomalies (array-like, [Radian], default=None):
                The true anomalies at which to illustrate the system. Provide either
                this or ``times`` but not both.
            orbit (bool, default=True):
                Whether to plot a trace of the planet's orbital path
            reflected (bool, default=False):
                Whether to color in the planet according to its reflected flux profile.
                Can optionally include this or ``emitted`` but not both.
            emitted (bool, default=False):
                Whether to color in the planet according to its emitted flux profile.
                Can optionally include this or ``reflected`` but not both.
            star_fill (bool, default=True):
                Whether to color in the star according to its limb darkening profile.
                Note that the lowest color contour is bounded at zero, so if you have an
                unphysical limb darkening law where some radii are negative, those
                will appear as gaps (most often the contours will not reach the black
                outline of the star, which is always drawn).
            window_size (float, [Rstar], default=0.4):
                The size of the plotting window. The window will be centered on the
                mean position of the planet across all of the suggested times, unless
                ``star_centered`` is set to ``True``.
            star_centered (bool, default=False):
                Whether to center the plot on the star rather than the planet.
            nsamples (int, default=50_000):
                The number of random samples of the planet's surface to draw when
                illustrating the system. A larger number will increase the resolution
                of the plot but result in longer computation times.
            figsize (tuple, default=(8, 8)):
                The size of the figure to display. Passed directly to
                ``matplotlib.pyplot.subplots``.

        Returns:
            None:
            This method is used for its side effects of displaying a plot, not for its
            return value.

        """

        if emitted:
            assert (
                reflected == False
            ), "Can't illustrate both reflected and emitted flux"
            assert self._state["parameterize_with_projected_ellipse"] == False, (
                "Can't illustrate emitted flux when only describing the 2D outline of"
                "the planet"
            )
        if reflected:
            assert emitted == False, "Can't illustrate both reflected and emitted flux"
            assert self._state["parameterize_with_projected_ellipse"] == False, (
                "Can't illustrate reflected flux when only describing the 2D outline of"
                "the planet"
            )

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        info = self._illustrate_helper(
            times=times, true_anomalies=true_anomalies, nsamples=nsamples
        )

        if star_centered:
            im_center_x = 0
            im_center_y = 0
        else:
            im_center_x = jnp.mean(info["planet_x_outlines"])
            im_center_y = jnp.mean(info["planet_y_outlines"])

        star = plt.Circle((0, 0), 1, color="black", fill=False)
        ax.add_artist(star)
        if star_fill & (not jnp.all(self._state["ld_u_coeffs"] == 0)):
            # lifted from engine.polynomial_limb_darkened_transit
            u_coeffs = jnp.ones(self._state["ld_u_coeffs"].shape[0] + 1) * (-1)
            u_coeffs = u_coeffs.at[1:].set(self._state["ld_u_coeffs"])
            g_coeffs = jnp.matmul(self._state["greens_basis_transform"], u_coeffs)
            normalization_constant = 1 / (
                jnp.pi * (g_coeffs[0] + (2 / 3) * g_coeffs[1])
            )

            def _star_radial_profile(r):
                us = jnp.ones(self._state["ld_u_coeffs"].shape[0] + 1) * (-1)
                us = us.at[1:].set(self._state["ld_u_coeffs"])
                mu = jnp.sqrt(1 - r**2)
                powers = jnp.arange(len(us))
                return -jnp.sum(us * (1 - mu) ** powers) * normalization_constant

            X = jnp.linspace(-1, 1, 300)
            Y = jnp.linspace(-1, 1, 300)
            X, Y = jnp.meshgrid(X, Y)
            R = jnp.sqrt(X**2 + Y**2)
            Z = jax.vmap(_star_radial_profile)(R.flatten()).reshape(X.shape)

            min_val = jnp.max(jnp.array([0, jnp.nanmin(Z)]))
            max_val = jnp.nanmax(Z)
            ax.contourf(
                X, Y, Z, cmap="copper", levels=jnp.linspace(min_val, max_val, 20)
            )
        elif star_fill:
            fill = plt.Circle((0, 0), 1, color="orange", fill=True, alpha=0.5)
            ax.add_artist(fill)

        if orbit:
            ax.plot(
                info["orbit_positions"][0, :],
                info["orbit_positions"][1, :],
                color="black",
                ls="--",
                lw=1,
                label="Orbit",
            )

        for i in range(len(info["planet_x_outlines"])):
            ax.plot(
                info["planet_x_outlines"][i],
                info["planet_y_outlines"][i],
                color="black",
                lw=1,
                label="Planet",
            )

            if reflected:
                ax.hexbin(
                    info["sample_xs"][i],
                    info["sample_ys"][i],
                    info["reflected_intensity"][i],
                    cmap="plasma",
                    gridsize=100,
                    mincnt=1,
                )

            if emitted:
                ax.hexbin(
                    info["sample_xs"][i],
                    info["sample_ys"][i],
                    info["emitted_intensity"][i],
                    cmap="plasma",
                    gridsize=100,
                    mincnt=1,
                )

        ax.set(
            aspect="equal",
            xlim=(im_center_x - window_size / 2, im_center_x + window_size / 2),
            ylim=(im_center_y - window_size / 2, im_center_y + window_size / 2),
        )

        return

    @staticmethod
    def fit_limb_darkening_profile(intensities, order=None, mus=None, rs=None):
        """
        Convert a stellar limb darkening profile to a polynomial representation.

        Given a set of stellar parameters, one can use a grid of stellar models to compute
        the limb darkening profile as a function of projected `r` or of
        `mu = sqrt(1 - r**2)`. These profiles are often then approximated with one of a few
        common limb darkening "laws", such as the quadratic or 4-parameter non-linear laws.
        Since `squishyplanet` only supports polynomial limb darkening profiles, but can
        support nearly arbitrary orders, we can approximate the profile with a polynomial.
        This is a convenience function for converting between a grid-derived profile and its
        best-fit polynomial representation in the correct basis for `squishyplanet`.

        Args:
            intensities (array-like):
                The relative intensities of the star at a given `mu` or `r`.
            order (int):
                The order of the polynomial to fit to the limb darkening profile. Note that
                in the `squishyplanet` basis, the polynomial is defined as
                `1 - sum_{i=1}^{order} u_i (1 - mu)^i`, so the number of coefficients is
                `order`, not `order+1`.
            mus (array-like, default=None):
                The `mu` values at which the intensities were computed. If `rs` is not
                provided, this is required.
            rs (array-like, default=None):
                The `r` values at which the intensities were computed. If `mus` is not
                provided, this is required.

        Returns:
            array-like:
                The coefficients of the polynomial representation of the limb darkening
                profile. These can then be used as the `ld_u_coeffs` parameter in
                `OblateSystem`.

        """
        return _fit_limb_darkening_profile(
            intensities=intensities, order=order, mus=mus, rs=rs
        )

    @staticmethod
    def limb_darkening_profile(ld_u_coeffs=None, r=None, mu=None):
        """
        Compute the limb darkening profile of the star at a given radius.

        Meant as a helper function for sanity checks and plotting, especially if you're
        using higher-order limb darkening laws and are concerned if the profile is
        positive/monotonic.

        Args:
            ld_u_coeffs (array-like, default=self.state["ld_u_coeffs"]):
                The coefficients of the polynomial limb darkening law.
            r (float or array-like, default=None):
                The radius at which to compute the limb darkening profile. Must be
                between 0 and 1. If provided, ``mu`` should be ``None``.
            mu (float or array-like, default=None):
                The cosine of the angle between the line of sight and the normal to the
                surface of the star. Must be between 0 and 1. If provided, ``r`` should
                be ``None``.

        Returns:
            Array:
                The limb darkening profile of the star at the given r or mu values.

        """
        assert (mu is None) != (r is None), "Only one of `mu` or `r` should be provided"

        if ld_u_coeffs is None:
            ld_u_coeffs = self._state["ld_u_coeffs"]
            greens_transform = self._state["greens_basis_transform"]
        else:
            greens_transform = generate_change_of_basis_matrix(len(ld_u_coeffs))

        if r is None:
            r = jnp.sqrt(1 - mu**2)

        u_coeffs = jnp.ones(ld_u_coeffs.shape[0] + 1) * (-1)
        u_coeffs = u_coeffs.at[1:].set(ld_u_coeffs)

        g_coeffs = jnp.matmul(greens_transform, u_coeffs)

        # total flux from the star. 1/eq. 28 in Agol, Luger, and Foreman-Mackey 2020
        normalization_constant = 1 / (jnp.pi * (g_coeffs[0] + (2 / 3) * g_coeffs[1]))

        def inner(r):
            us = jnp.ones(ld_u_coeffs.shape[0] + 1) * (-1)
            us = us.at[1:].set(ld_u_coeffs)
            mu = jnp.sqrt(1 - r**2)
            powers = jnp.arange(len(us))
            return -jnp.sum(us * (1 - mu) ** powers) * normalization_constant

        if (type(r) == float) | (type(r) == int):
            return inner(r)
        else:
            return jax.vmap(inner)(r)

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
        return self._lightcurve_fwd_grad_enforced(params)

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
        return self._loglike_fwd_grad_enforced(params)


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ),
)
def _lightcurve(
    params,
    tidally_locked,
    compute_reflected_phase_curve,
    compute_emitted_phase_curve,
    compute_stellar_ellipsoidal_variations,
    compute_stellar_doppler_variations,
    parameterize_with_projected_ellipse,
    oversample,
    random_seed,
    phase_curve_nsamples,
    extended_illumination_npts,
    state,
):
    # always compute the primary transit and trend
    for key in params.keys():
        state[key] = params[key]
    transit = poly_lightcurve(state, parameterize_with_projected_ellipse)
    trend = jnp.polyval(state["systematic_trend_coeffs"], state["times"])

    # if you don't want any phase curve stuff, you're done
    if (not compute_reflected_phase_curve) & (not compute_emitted_phase_curve) and (
        not compute_stellar_doppler_variations
    ) & (not compute_stellar_ellipsoidal_variations):
        oversampled_curve = transit + trend
        if oversample > 1:
            c = (
                oversampled_curve.reshape(-1, oversample) * state["stencil"][None, :]
            ).sum(axis=1)
        else:
            c = oversampled_curve
        return c

    ######################################################
    # compute the planet's contribution to the phase curve
    ######################################################

    # generate the radii and thetas that you'll reuse at each timestep
    sample_radii, sample_thetas = generate_sample_radii_thetas(
        jax.random.key(random_seed),
        jnp.arange(phase_curve_nsamples),
    )

    # solve Kepler's equation to get the true anomalies
    time_deltas = state["times"] - state["t_peri"]
    mean_anomalies = 2 * jnp.pi * time_deltas / state["period"]
    true_anomalies = kepler(mean_anomalies, state["e"])
    state["f"] = true_anomalies
    if tidally_locked:
        state["prec"] = state["f"]

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
        if extended_illumination_npts == 1:
            reflected = reflected_phase_curve(
                sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
            )
        else:
            offsets = extended_illumination_offsets(**state)
            three = planet_3d_coeffs_extended_illumination(**state, offsets=offsets)
            two = planet_2d_coeffs(**three)
            reflected = extended_illumination_reflected_phase_curve(
                sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c, offsets
            )
        emitted = 0.0

    # just the emitted component
    elif (not compute_reflected_phase_curve) & compute_emitted_phase_curve:
        reflected = 0.0
        emitted = emission_phase_curve(sample_radii, sample_thetas, two, three, state)

    # both reflected and emitted components. this function shares some of the
    # computation between the two, so it's a bit faster than running them separately
    elif (
        compute_reflected_phase_curve
        & compute_emitted_phase_curve
        & (extended_illumination_npts == 1)
    ):
        reflected, emitted = phase_curve(
            sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
        )

    elif (
        compute_reflected_phase_curve
        & compute_emitted_phase_curve
        & (extended_illumination_npts != 1)
    ):
        if extended_illumination_npts == 1:
            reflected = reflected_phase_curve(
                sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
            )
        else:
            offsets = extended_illumination_offsets(**state)
            three = planet_3d_coeffs_extended_illumination(**state, offsets=offsets)
            two = planet_2d_coeffs(**three)
            reflected = extended_illumination_reflected_phase_curve(
                sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c, offsets
            )
        emitted = emission_phase_curve(sample_radii, sample_thetas, two, three, state)
    else:
        reflected = 0.0
        emitted = 0.0

    ####################################################
    # compute the star's contribution to the phase curve
    ####################################################

    if compute_stellar_ellipsoidal_variations:
        ellipsoidal = stellar_ellipsoidal_variations(
            state["f"], state["stellar_ellipsoidal_alpha"], state["period"]
        )
    else:
        ellipsoidal = 0.0

    if compute_stellar_doppler_variations:
        doppler = stellar_doppler_variations(
            state["f"], state["stellar_doppler_alpha"], state["period"]
        )
    else:
        doppler = 0.0

    ####################################################
    # put it all together
    ####################################################

    oversampled_curve = transit + trend + reflected + emitted + ellipsoidal + doppler
    if oversample > 1:
        c = (oversampled_curve.reshape(-1, oversample) * state["stencil"][None, :]).sum(
            axis=1
        )
    else:
        c = oversampled_curve
    return c


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ),
)
def _loglike(
    params,
    tidally_locked,
    compute_reflected_phase_curve,
    compute_emitted_phase_curve,
    compute_stellar_ellipsoidal_variations,
    compute_stellar_doppler_variations,
    parameterize_with_projected_ellipse,
    oversample,
    random_seed,
    phase_curve_nsamples,
    extended_illumination_npts,
    state,
):
    lc = _lightcurve(
        params,
        tidally_locked,
        compute_reflected_phase_curve,
        compute_emitted_phase_curve,
        compute_stellar_ellipsoidal_variations,
        compute_stellar_doppler_variations,
        parameterize_with_projected_ellipse,
        oversample,
        random_seed,
        phase_curve_nsamples,
        extended_illumination_npts,
        state,
    )

    for key in params.keys():
        state[key] = params[key]

    resids = state["data"] - lc
    var = jnp.exp(state["log_jitter"]) ** 2 + state["uncertainties"] ** 2

    return jnp.sum(-0.5 * (resids**2 / var + jnp.log(2 * jnp.pi * var)))


@partial(jax.jit, static_argnums=(1,))
def _fit_limb_darkening_profile(intensities, order=None, mus=None, rs=None):
    """
    Convert a stellar limb darkening profile to a polynomial representation.

    Given a set of stellar parameters, one can use a grid of stellar models to compute
    the limb darkening profile as a function of projected `r` or of
    `mu = sqrt(1 - r**2)`. These profiles are often then approximated with one of a few
    common limb darkening "laws", such as the quadratic or 4-parameter non-linear laws.
    Since `squishyplanet` only supports polynomial limb darkening profiles, but can
    support nearly arbitrary orders, we can approximate the profile with a polynomial.
    This is a convenience function for converting between a grid-derived profile and its
    best-fit polynomial representation in the correct basis for `squishyplanet`.

    Args:
        intensities (array-like):
            The relative intensities of the star at a given `mu` or `r`.
        order (int):
            The order of the polynomial to fit to the limb darkening profile. Note that
            in the `squishyplanet` basis, the polynomial is defined as
            `1 - sum_{i=1}^{order} u_i (1 - mu)^i`, so the number of coefficients is
            `order`, not `order+1`.
        mus (array-like, default=None):
            The `mu` values at which the intensities were computed. If `rs` is not
            provided, this is required.
        rs (array-like, default=None):
            The `r` values at which the intensities were computed. If `mus` is not
            provided, this is required.

    Returns:
        array-like:
            The coefficients of the polynomial representation of the limb darkening
            profile. These can then be used as the `ld_u_coeffs` parameter in
            `OblateSystem`.

    """
    if rs is not None:
        mus = jnp.sqrt(1 - rs**2)
    powers = jnp.arange(order + 1)[1:]
    a = ((1 - mus) ** powers[:, None]).T
    b = intensities - 1
    ld_u_coeffs = -jnp.linalg.lstsq(a, b)[0]
    return ld_u_coeffs
