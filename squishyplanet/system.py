import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bulge.engine.planet_3d import planet_3d_coeffs
from bulge.engine.planet_2d import planet_2d_coeffs
from bulge.engine.parametric_ellipse import poly_to_parametric
from bulge.engine.greens_basis_transform import generate_change_of_basis_matrix
from bulge.engine.kepler import kepler, skypos
from bulge.engine.polynomial_limb_darkened_transit import lightcurve
from bulge.engine.phase_curve_utils import (
    generate_sample_radii_thetas,
    sample_surface,
    planet_surface_normal,
    surface_star_cos_angle,
    lambertian_reflection,
    pre_squish_transform,
    emission_profile,
    reflected_phase_curve,
    emission_phase_curve,
    phase_curve

)

import copy
from functools import partial
import pprint


class OblateSystem:
    def __init__(
        self,
        times=None,
        t_peri=None,
        period=None,
        a=None,
        e=0.0,
        i=jnp.pi/2,
        Omega=jnp.pi,
        omega=0.0,
        obliq=0.0,
        prec=0.0,
        r=None,
        f1=0.0,
        f2=0.0,
        ld_u_coeffs=jnp.array([0, 0]),
        hotspot_latitude=0.0,
        hotspot_longitude=0.0,
        hotspot_concentration=0.2,
        reflected_albedo=1.0,
        emitted_scale=1.0,
        systematic_offset=0.0,
        systematic_linear=0.0,
        log_jitter=-10,
        tidally_locked=True,
        compute_reflected_phase_curve=False,
        compute_emitted_phase_curve=False,
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
            "systematic_offset",
            "systematic_linear",
            "log_jitter",
            "tidally_locked",
            "compute_reflected_phase_curve",
            "compute_emitted_phase_curve",
            "phase_curve_nsamples",
            "random_seed",
            "data",
            "uncertainties"
        ]

        state = {}
        for key in state_keys:
            state[key] = locals()[key]
        self._state = state


        self.validate_inputs()

        # necessary for all light curves
        self._state["greens_basis_transform"] = (
            generate_change_of_basis_matrix(len(self._state["ld_u_coeffs"]))
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

    def validate_inputs(self):
        for key, val in self._state.items():
            if type(val) ==  type(None):
                raise ValueError(f"'{key}' is a required parameter")

        self._state["ld_u_coeffs"] = jnp.array(self._state["ld_u_coeffs"])
        assert self._state["ld_u_coeffs"].shape[0] >= 2, "ld_u_coeffs must have at least 2 (even if higher-order terms are 0)"
        assert type(self._state["phase_curve_nsamples"]) == int, "phase_curve_nsamples must be an integer"
        assert type(self._state["random_seed"]) == int, "random_seed must be an integer"

        if self._state["e"] == 0:
            assert (
                self._state["omega"] == 0
            ), "omega must be 0 for a circular orbit"

        shapes = []
        for key in self._state.keys():
            if (key == "ld_u_coeffs") | (key == "phase_curve_nsamples") | (key == "random_seed"):
                continue
            elif type(self._state[key]) == bool:
                continue

            if (type(self._state[key]) == float) | (
                type(self._state[key]) == int
            ):
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

    def _illustrate_helper(
        self, times=None, true_anomalies=None, nsamples=50_000
    ):

        if (times is not None) & (true_anomalies is not None):
            raise ValueError(
                "Provide either times or true anomalies but not both"
            )

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
        return _lightcurve(
            compute_reflected_phase_curve=self._state["compute_reflected_phase_curve"],
            compute_emitted_phase_curve=self._state["compute_emitted_phase_curve"],
            random_seed=self._state["random_seed"],
            phase_curve_nsamples=self._state["phase_curve_nsamples"],
            state=self._state,
            params=params,
        )

    def loglike(self, params={}):
        return _loglike(
            compute_reflected_phase_curve=self._state["compute_reflected_phase_curve"],
            compute_emitted_phase_curve=self._state["compute_emitted_phase_curve"],
            random_seed=self._state["random_seed"],
            phase_curve_nsamples=self._state["phase_curve_nsamples"],
            state=self._state,
            params=params,
        )

    @property
    def state(self):
        return self._state

    def __repr__(self):
        s = pprint.pformat(self.state)
        return f"OblateSystem(\n{s}\n)"


@partial(
    jax.jit,
    static_argnums=(
        0,
        1,
        2,
        3,
    ),
)
def _lightcurve(
    compute_reflected_phase_curve,
    compute_emitted_phase_curve,
    random_seed,
    phase_curve_nsamples,
    state,
    params,
):

    # if all you want is the transit, don't do any of the phase curve calculations
    if (not compute_reflected_phase_curve) & (not compute_emitted_phase_curve):
        for key in params.keys():
            state[key] = params[key]
        trend = state["systematic_offset"] + state["systematic_linear"] * state["times"]
        return lightcurve(state) + trend

    # if you do want a phase curve, generate the radii and thetas that you'll reuse
    # at each timestep
    sample_radii, sample_thetas = generate_sample_radii_thetas(
        jax.random.key(random_seed), jnp.arange(phase_curve_nsamples)
    )

    # just the reflected component
    if compute_reflected_phase_curve & (not compute_emitted_phase_curve):
        for key in params.keys():
            state[key] = params[key]

        transit = lightcurve(state)

        # technically these are all calculated in "transit", but since phase
        # curves are a) rare and b) expensive, we'll just do it again to keep
        # the transit section of the code more self-contained
        three = planet_3d_coeffs(**state)
        two = planet_2d_coeffs(**three)
        positions = skypos(**state)
        x_c = positions[0, :]
        y_c = positions[1, :]
        z_c = positions[2, :]
        reflected = reflected_phase_curve(
            sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
        )

        trend = state["systematic_offset"] + state["systematic_linear"] * state["times"]
        return transit + reflected + trend

    # just the emitted component
    elif (not compute_reflected_phase_curve) & compute_emitted_phase_curve:
        for key in params.keys():
            state[key] = params[key]
        transit = lightcurve(state)

        # technically these are all calculated in "transit", but since phase
        # curves are a) rare and b) expensive, we'll just do it again to keep
        # the transit section of the code more self-contained
        three = planet_3d_coeffs(**state)
        two = planet_2d_coeffs(**three)
        positions = skypos(**state)
        x_c = positions[0, :]
        y_c = positions[1, :]
        z_c = positions[2, :]
        emitted = emission_phase_curve(
            sample_radii, sample_thetas, two, three, state
        )

        trend = state["systematic_offset"] + state["systematic_linear"] * state["times"]
        return transit + emitted + trend

    # both reflected and emitted components
    elif compute_reflected_phase_curve & compute_emitted_phase_curve:
        for key in params.keys():
            state[key] = params[key]
        transit = lightcurve(state)

        # technically these are all calculated in "transit", but since phase
        # curves are a) rare and b) expensive, we'll just do it again to keep
        # the transit section of the code more self-contained
        three = planet_3d_coeffs(**state)
        two = planet_2d_coeffs(**three)
        positions = skypos(**state)
        x_c = positions[0, :]
        y_c = positions[1, :]
        z_c = positions[2, :]
        reflected, emitted = phase_curve(
            sample_radii, sample_thetas, two, three, state, x_c, y_c, z_c
        )

        trend = state["systematic_offset"] + state["systematic_linear"] * (state["times"])
        return transit + reflected + emitted + trend


@partial(
    jax.jit,
    static_argnums=(
        0,
        1,
        2,
        3,
    ),
)
def _loglike(
    compute_reflected_phase_curve,
    compute_emitted_phase_curve,
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
    
    return jnp.sum(-0.5 * (resids ** 2 / var + jnp.log(var)))
