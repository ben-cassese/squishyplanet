Changelog
++++++++++

**0.1.2 (05/2024)**

Fixed issue #7. Previously, calls to ``OblateSystem.lightcurve()`` after initializing the ``OblateSystem`` object were not recalculating true anomalies even if the ``period`` and/or ``t_peri`` changed.

**0.1.1 (05/2024)**

Initial JOSS submission version, fixed issues #1 and #5.