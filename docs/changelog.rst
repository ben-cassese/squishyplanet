Changelog
++++++++++

.. **0.2.0 (08/2024)**

.. Addressing JOSS review comments:

.. - placeholder

.. Other changes:

.. - When dealing with a projected 2D ellipse (i.e., not a tidally locked planet) ``projected_r`` has been replaced with ``projected_effective_r``. Previously there was a strong degeneracy between ``projected_r`` and ``projected_f``, since the combination of the two dictated the area of the ellipse, and therefore transit depth, which fits are much more sensitive to than to slight deviations to ingress/egress shape. ``projected_effective_r`` is the radius of a circle with the same area as the ellipse, and is therefore a more physically meaningful parameter to fit for. 


**0.1.2 (05/2024)**

Fixed issue `#7
<https://github.com/ben-cassese/squishyplanet/issues/7/>`_. Previously, calls to ``OblateSystem.lightcurve()`` after initializing the ``OblateSystem`` object were not recalculating true anomalies even if the ``period`` and/or ``t_peri`` changed.

**0.1.1 (05/2024)**

Initial JOSS submission version, fixed issues #1 and #5.