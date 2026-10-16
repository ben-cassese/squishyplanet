Changelog
++++++++++

**0.6.0**

- Fixed the projected-ellipse orientation when the quadratic coefficients along
  the two sky axes are equal. Tilted ellipses at 45 degrees now retain their
  orientation instead of being treated as axis-aligned.
- Fixed JAX derivatives at zero flattening in both the projected 2D and physical
  3D planet parameterizations. The transit engine now constructs the boundary
  directly from the shape parameters, avoiding the undefined major-axis angle of
  a circle. Added regression tests for circular-limit gradients.
- Replaced SymPy-based construction of the limb-darkening change-of-basis matrix
  with a NumPy linear solve. SymPy is now a test-only dependency, with regression
  comparisons against the original symbolic calculation at orders up to 50.
- Removed the remaining unused phase-curve and extended-illumination internals.
  The surface-sampling and illumination helpers used by ``illustrate`` now live
  in ``engine.illustrate_utils``; direct imports from the removed
  ``engine.phase_curve_utils`` module must be updated.
- Consolidated repeated limb-darkening profile and illustration code, and
  simplified the 3D-to-2D projection helpers without changing their formulas.
- Reduced runtime dependencies to JAX, NumPy, and Matplotlib. Removed quadax,
  moved Astropy and tqdm to the test dependency group, and removed unused tox
  tooling. JAX is now declared directly as a dependency.
- Updated integration-method documentation and lint compatibility. CI now avoids
  duplicate push and pull-request runs and cancels superseded pull-request runs.

**0.5.0**

- Added ``RingedSystem``, a subclass of ``OblateSystem`` for planets hosting an
  opaque, flat, circular ring bounded by an inner and outer radius. Rings lie in
  the planet's equatorial plane by default, with options for independent
  orientation. Blocked flux is computed by inclusion-exclusion of overlapping
  regions using Green's-theorem boundary integrals. ``RingedSystem.illustrate``
  overlays the projected ring edges and a filled annulus.
- Generalized the transit engine's limb-crossing logic to split the planet
  outline and stellar limb at all crossings, including four-intersection cases.
  This fixes light curves for long, thin projected ellipses crossing the star,
  including nearly edge-on rings and extreme projected planet shapes.
- Added ring geometry, flux, and gradient tests, with independent Monte Carlo
  and Shapely comparisons, plus stress tests for grazing transits, tangencies,
  extreme flattening, and higher-order limb darkening.
- Expanded type annotations and updated tutorials, including examples for ringed
  planets and performance comparisons.

**0.4.0**

- **Breaking:** Removed ``OblateSystem.loglike``. Users now compute their own
  likelihood by comparing ``OblateSystem.lightcurve`` with their data. Removed
  the constructor arguments ``data``, ``uncertainties``, ``log_jitter``, and
  ``systematic_trend_coeffs``. The transit-fitting tutorial demonstrates this
  pattern.
- **Breaking:** Removed phase-curve and stellar-variation calculations from
  ``OblateSystem``, together with ``compute_reflected_phase_curve``,
  ``compute_emitted_phase_curve``, ``compute_stellar_ellipsoidal_variations``,
  ``compute_stellar_doppler_variations``, ``phase_curve_nsamples``,
  ``extended_illumination_npts``, and ``random_seed``.
  ``OblateSystem.illustrate`` retains its ``reflected`` visualization option.
- Replaced adaptive quadax integration with fixed-grid quadrature: a
  reparameterized Gauss-Legendre rule for partial transits and a periodic
  trapezoid rule for full transits. Added numerical stability and performance
  improvements to the transit engine.
- Removed the separate Fortran implementation.
- Dropped Python 3.9 support, adopted uv for dependency management, and added
  pre-commit linting, formatting, and spelling checks.

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
