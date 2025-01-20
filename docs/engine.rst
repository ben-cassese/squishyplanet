Engine
======

``squishyplanet`` was designed to expose :func:`OblateSystem` to the user, while keeping
most of the actual computation in an "engine" directory. Modules included here are meant
to rely on as few libraries as possible aside from basic jax.numpy, and are meant to
contain functions that are entirely ``jit``-able. Actually calling these functions may
be helpful for building a more complex likelihood function (e.g., if you want to jointly
fit multiple spectral channels, or are building a more sophisticated emission model),
but in general we assume most interaction will be through the :func:`OblateSystem`
class.

The code itself is based heavily on `Agol, Luger, and Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_,
and the documentation makes references to specific equations in that paper. Those going
through this code are encouraged to review that paper and be familiar with the Green's
basis tranformation used to convert 2D surface integrals to 1D line integrals.

-----

.. automodule:: polynomial_limb_darkened_transit
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: planet_3d
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: planet_2d
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: phase_curve_utils
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: parametric_ellipse
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: kepler
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: greens_basis_transform
    :members:
    :undoc-members:
    :show-inheritance:
