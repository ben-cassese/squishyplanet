.. _RingedSystem:

RingedSystem
============

A subclass of :func:`OblateSystem` for planets that host an opaque, flat, circular
ring, bounded by an inner and an outer radius and (by default) lying in the planet's
equatorial plane. The blocked flux is computed with an exact inclusion-exclusion
decomposition into intersections of convex regions, each evaluated with the same
Green's-theorem boundary integrals as the planet-only model; see
``engine.ringed_transit`` for the details.


.. automodule:: ringed_system
    :members:
    :undoc-members:
    :show-inheritance:
