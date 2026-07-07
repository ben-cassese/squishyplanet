import jax

jax.config.update("jax_enable_x64", True)
import pprint
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from squishyplanet.engine.kepler import kepler, t0_to_t_peri
from squishyplanet.engine.ringed_transit import ringed_lightcurve
from squishyplanet.engine.rings import ring_para_coeffs
from squishyplanet.oblate_system import OblateSystem


class RingedSystem(OblateSystem):
    """A :class:`OblateSystem` whose planet hosts an opaque, flat, circular ring.

    The ring is assumed to be perfectly flat, perfectly circular, perfectly opaque,
    concentric with the planet, and bounded by an inner and an outer radius. By
    default it lies in the planet's equatorial plane (its 3D orientation set by the
    planet's ``obliq`` and ``prec``, including the tidally-locked time dependence);
    passing ``ring_obliq`` and ``ring_prec`` explicitly decouples the ring's
    orientation from the planet's.

    All other arguments, defaults, and methods match :class:`OblateSystem`, except
    that rings require the full 3D parameterization
    (``parameterize_with_projected_ellipse`` must remain ``False``).

    Args:
        ring_inner_r (float, [Rstar], default=None):
            The inner radius of the ring. Must be larger than the planet's equatorial
            radius ``r``. A required parameter, will raise an error if not provided.
        ring_outer_r (float, [Rstar], default=None):
            The outer radius of the ring. Must be larger than ``ring_inner_r``; note
            that we ignore the physics of ring formation and stability, so it is not
            capped at anything like the Roche radius. A required parameter, will raise
            an error if not provided.
        ring_obliq (float, [Radian], default=None):
            The obliquity of the ring plane, defined identically to the planet's
            ``obliq``. Provide both ``ring_obliq`` and ``ring_prec`` or neither; if
            neither, the ring tracks the planet's equatorial plane.
        ring_prec (float, [Radian], default=None):
            The precession angle of the ring plane, defined identically to the
            planet's ``prec``.
        **kwargs:
            All :class:`OblateSystem` arguments.

    """

    def __init__(
        self,
        ring_inner_r: float | None = None,
        ring_outer_r: float | None = None,
        ring_obliq: float | None = None,
        ring_prec: float | None = None,
        **kwargs: object,
    ) -> None:
        assert ring_inner_r is not None, "'ring_inner_r' is a required parameter"
        assert ring_outer_r is not None, "'ring_outer_r' is a required parameter"
        assert not kwargs.get("parameterize_with_projected_ellipse", False), (
            "rings require the full 3D parameterization: "
            "parameterize_with_projected_ellipse must be False"
        )
        assert (ring_obliq is None) == (ring_prec is None), (
            "provide both 'ring_obliq' and 'ring_prec' to decouple the ring from the "
            "planet's equatorial plane, or neither to keep them aligned"
        )

        super().__init__(**kwargs)

        assert ring_outer_r > ring_inner_r > 0, "need ring_outer_r > ring_inner_r > 0"
        assert ring_inner_r > float(jnp.asarray(self._state["r"]).ravel()[0]), (
            "ring_inner_r must be larger than the planet's equatorial radius r "
            "(the largest of its three axes)"
        )

        ring_tracks_planet = ring_obliq is None
        if ring_tracks_planet:
            # placeholders only: when tracking, the engine reads the planet's
            # obliq/prec (and the tidally-locked true anomaly) at call time
            ring_obliq = 0.0
            ring_prec = 0.0

        self._state["ring_inner_r"] = jnp.array([float(ring_inner_r)])
        self._state["ring_outer_r"] = jnp.array([float(ring_outer_r)])
        self._state["ring_obliq"] = jnp.array([float(ring_obliq)])
        self._state["ring_prec"] = jnp.array([float(ring_prec)])
        self._state["ring_tracks_planet"] = ring_tracks_planet

        # the parent's __init__ already built a jitted lightcurve, but its frozen
        # state snapshot predates the ring keys; rebuild it now that they exist
        self._lightcurve_fwd_grad_enforced = self._setup_lightcurve_func()

    def __repr__(self) -> str:
        s = pprint.pformat(self.state)
        return f"RingedSystem(\n{s}\n)"

    def _setup_lightcurve_func(self) -> Callable:
        constants = {
            "oversample": self._state["oversample"],
            "state": self._state,
        }

        frozen = jax.tree_util.Partial(_ringed_lightcurve, **constants)

        @jax.custom_vjp
        def lightcurve(params: dict) -> jax.Array:
            return frozen(params)

        def lightcurve_fwd(params: dict) -> tuple:
            output = frozen(params)
            jac = jax.jacfwd(frozen)(params)
            return output, (jac,)

        def lightcurve_bwd(res: tuple, g: jax.Array) -> tuple:
            jac = res
            val = jax.tree.map(lambda x: x.T @ g, jac)
            return val

        lightcurve.defvjp(lightcurve_fwd, lightcurve_bwd)
        lightcurve = jax.jit(lightcurve)

        return lightcurve

    def _ring_edge_outlines(self, true_anomalies: jax.Array) -> tuple:
        """Sky-plane outlines of both ring edges at the given true anomalies.

        Args:
            true_anomalies (Array): True anomalies at which to project the ring.

        Returns:
            Tuple:
                Arrays ``(outer_x, outer_y, inner_x, inner_y)``, each of shape
                ``(len(true_anomalies), 200)``.

        """
        thetas = jnp.linspace(0, 2 * jnp.pi, 200)
        outer_x, outer_y, inner_x, inner_y = [], [], [], []
        for f in true_anomalies:
            f = jnp.array([f])
            if self._state["ring_tracks_planet"]:
                ring_obliq = self._state["obliq"]
                ring_prec = f if self._state["tidally_locked"] else self._state["prec"]
            else:
                ring_obliq = self._state["ring_obliq"]
                ring_prec = self._state["ring_prec"]

            shared = dict(
                a=self._state["a"],
                e=self._state["e"],
                f=f,
                Omega=self._state["Omega"],
                i=self._state["i"],
                omega=self._state["omega"],
                ring_obliq=ring_obliq,
                ring_prec=ring_prec,
            )
            for rRing, xs, ys in (
                (self._state["ring_outer_r"], outer_x, outer_y),
                (self._state["ring_inner_r"], inner_x, inner_y),
            ):
                para = ring_para_coeffs(rRing=rRing, **shared)
                xs.append(
                    para["c_x1"] * jnp.cos(thetas)
                    + para["c_x2"] * jnp.sin(thetas)
                    + para["c_x3"]
                )
                ys.append(
                    para["c_y1"] * jnp.cos(thetas)
                    + para["c_y2"] * jnp.sin(thetas)
                    + para["c_y3"]
                )
        return (
            jnp.array(outer_x),
            jnp.array(outer_y),
            jnp.array(inner_x),
            jnp.array(inner_y),
        )

    def illustrate(
        self,
        times: jax.Array | None = None,
        true_anomalies: jax.Array | None = None,
        ring_fill: bool = True,
        **kwargs: object,
    ) -> None:
        """Visualize the layout of the system, including the ring, at one or more times.

        Identical to :func:`OblateSystem.illustrate` but overlays the projected
        outlines of both ring edges and (optionally) fills the annulus between them.
        Note that the layering is cosmetic: the filled annulus is drawn on top of the
        planet even though half of the physical ring passes behind it.

        Args:
            times (array-like, [Days], default=None):
                See :func:`OblateSystem.illustrate`.
            true_anomalies (array-like, [Radian], default=None):
                See :func:`OblateSystem.illustrate`.
            ring_fill (bool, default=True):
                Whether to fill the annulus between the ring edges, in addition to
                drawing their outlines.
            **kwargs:
                All other :func:`OblateSystem.illustrate` arguments.

        Returns:
            None:
            This method is used for its side effects of displaying a plot, not for its
            return value.

        """
        super().illustrate(times=times, true_anomalies=true_anomalies, **kwargs)
        ax = plt.gca()

        # resolve times/true anomalies the same way the parent helper does
        if (times is not None) & (true_anomalies is not None):
            raise ValueError("Provide either times or true anomalies but not both")
        if times is not None:
            t_peri = self._state.get("t_peri", None)
            if t_peri is None:
                t_peri = t0_to_t_peri(
                    e=self._state["e"],
                    i=self._state["i"],
                    omega=self._state["omega"],
                    period=self._state["period"],
                    t0=self._state["t0"],
                )
            time_deltas = times - t_peri
            mean_anomalies = 2 * jnp.pi * time_deltas / self._state["period"]
            true_anomalies = kepler(mean_anomalies, self._state["e"])
        elif true_anomalies is None:
            true_anomalies = jnp.array([jnp.pi / 2])
        if isinstance(true_anomalies, float | int):
            true_anomalies = jnp.array([true_anomalies])

        outer_x, outer_y, inner_x, inner_y = self._ring_edge_outlines(true_anomalies)

        for i in range(len(true_anomalies)):
            ax.plot(outer_x[i], outer_y[i], color="black", lw=1, label="Ring")
            ax.plot(inner_x[i], inner_y[i], color="black", lw=1)
            if ring_fill:
                # a two-subpath (outer + reversed inner) closed path fills as an
                # annulus under matplotlib's fill rule
                verts = jnp.concatenate(
                    (
                        jnp.stack((outer_x[i], outer_y[i]), axis=1),
                        jnp.stack((inner_x[i], inner_y[i]), axis=1)[::-1],
                    )
                )
                codes = (
                    [Path.MOVETO]
                    + [Path.LINETO] * (outer_x[i].shape[0] - 1)
                    + [Path.MOVETO]
                    + [Path.LINETO] * (inner_x[i].shape[0] - 1)
                )
                patch = PathPatch(
                    Path(verts, codes), facecolor="gray", edgecolor="none", alpha=0.5
                )
                ax.add_patch(patch)

        return


@partial(jax.jit, static_argnums=(1,))
def _ringed_lightcurve(params: dict, oversample: int, state: dict) -> jax.Array:
    for key in params:
        state[key] = params[key]
    transit = ringed_lightcurve(state)
    if oversample > 1:
        c = (transit.reshape(-1, oversample) * state["stencil"][None, :]).sum(axis=1)
    else:
        c = transit
    return c
