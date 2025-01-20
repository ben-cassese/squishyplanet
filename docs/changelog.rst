Changelog
++++++++++

**0.3.1 (10/2024)**

- Enforced forward-mode differentiation under the hood in ``OblateSystem`` to speed up gradient evaluation for ``OblateSystem.loglike`` or any functions that use ``OblateSystem.lightcurve``.

**0.3.0 (10/2024)**

- Added ``limb_darkening_laws`` module, which contains convenience functions for transforming limb darkening coefficients of commonly used limb darkening laws into coefficients of high-order polynomials that ``squishyplanet`` can use to model the intensity profile of the star.

**0.2.2 (09/2024)**

- Added ability to parameterize the planet's orbit via t0, the time of transit center, instead of t_peri. This is useful for fitting systems with non-zero eccentricity.

**0.2.1 (09/2024)**

- Fixes issues `#16 <https://github.com/ben-cassese/squishyplanet/issues/16>`_ and `#17 <https://github.com/ben-cassese/squishyplanet/issues/17>`_.

**0.2.0 (08/2024)**

- The ``log_jitter`` term now represents the *standard deviation* of the added noise, not the variance. This brings it in line with the definition of measurement uncertainties.
- When dealing with a projected 2D ellipse (i.e., not a tidally locked planet) ``projected_r`` has been replaced with ``projected_effective_r``. Previously there was a strong degeneracy between ``projected_r`` and ``projected_f``, since the combination of the two dictated the area of the ellipse, and therefore transit depth, which fits are much more sensitive to than to slight deviations to ingress/egress shape. ``projected_effective_r`` is the radius of a circle with the same area as the ellipse, and is therefore a more physically meaningful parameter to fit for.
- Added fit_limb_darkening_profile() as a convenience function for approximating limb darkening profiles computed via stellar grids as high-order polynomial laws that ``squishyplanet`` can use.
- ``tidally_locked`` is now a required parameter when creating an ``OblateSystem`` object, which is a change from its previous default of True.


**0.1.2 (05/2024)**

Fixed issue `#7
<https://github.com/ben-cassese/squishyplanet/issues/7/>`_. Previously, calls to ``OblateSystem.lightcurve()`` after initializing the ``OblateSystem`` object were not recalculating true anomalies even if the ``period`` and/or ``t_peri`` changed.

**0.1.1 (05/2024)**

Initial JOSS submission version, fixed issues #1 and #5.
