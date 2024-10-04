import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pprint

from squishyplanet import OblateSystem


class RingedSystem(OblateSystem):
    """
    A class for modeling a (potentially oblate or triaxial) planet with a ring.

    This inherits from OblateSystem, so has all of the same input parameters, plus those
    for the ring.

    Args:
        ring_inner_r (float, [Rstar], default=None):
            The inner edge of the ring, in units of the star's radius. A required
            parameter, will raise an error if not provided.
        ring_outer_r (float, [Rstar], default=None):
            The outer edge of the ring, in units of the star's radius. A required
            parameter, will raise an error if not provided.
        ring_obliq (float, [radians], default=None):
            The obliquity of the ring. If it and `ring_prec` are not provided, the ring
            will be assumed to be in the plane of the planet's equator.
        ring_prec (float, [radians], default=None):
            The precession of the ring. If it and `ring_obliq` are not provided, the
            ring will be assumed to be in the plane of the planet's equator.

    """

    def __init__(
        self,
        ring_inner_r=None,
        ring_outer_r=None,
        ring_obliq=None,
        ring_prec=None,
        **kwargs,
    ):
        k = kwargs.copy()
        k.pop("ring_inner_r", None)
        k.pop("ring_outer_r", None)
        k.pop("ring_obliq", None)
        k.pop("ring_prec", None)

        super().__init__(**k)

        assert ring_inner_r is not None, "ring_inner_r must be specified"
        assert ring_outer_r is not None, "ring_outer_r must be specified"
        assert (ring_obliq is None and ring_prec is None) or (
            ring_obliq is not None and ring_prec is not None
        ), "both ring_obliq and ring_prec must be provided, or neither"
        assert (
            ring_inner_r < ring_outer_r
        ), "ring_inner_r must be smaller than ring_outer_r"
        assert (
            ring_inner_r > self._state["r"]
        ), "ring_inner_r must be larger than the planet's radius"

        self._state["ring_inner_r"] = ring_inner_r
        self._state["ring_outer_r"] = ring_outer_r
        if ring_obliq is None and ring_prec is None:
            ring_obliq = self._state["obliq"]
            ring_prec = self._state["prec"]
        self._state["ring_obliq"] = ring_obliq
        self._state["ring_prec"] = ring_prec

    def __repr__(self):
        s = pprint.pformat(self.state)
        return f"RingedSystem(\n{s}\n)"

    def illustrate(self):
        pass

    def lightcurve(self):
        pass
