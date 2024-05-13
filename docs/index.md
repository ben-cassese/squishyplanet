<!-- .. squishyplanet documentation master file, created by
   sphinx-quickstart on Mon Apr 15 08:10:41 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. -->

squishyplanet 
=============

<div align="center"> <img src="./_static/media/videos/_static/480p15/Banner_ManimCE_v0.17.3.gif" width="100%"> </div>


<br>Welcome to the documentation for `squishyplanet`! This lightweight package is designed to produce realistic lightcurves and phase curves of non-spherical exoplanets.

Most of the time, assuming that an exoplanet is a perfect sphere is a great approximation. However, in cases where we both a) expect the planet to be slightly deformed (either through gravitational interaction with its host star or through its own rapid rotation) and b) have high-precision data, fitting for its triaxial shape can provide additional constraints on the planet's interior properties and evolution. ``squishyplanet`` can generate models of these triaxial planets, then leaves the choice of inference framework up to you.

In the limiting case where the planet is forced to be spherical, `squishyplanet` is designed to be as accurate as [jaxoplanet](https://jax.exoplanet.codes/en/latest/) and [starry](https://starry.readthedocs.io/en/latest/) (see [Compare with jaxoplanet notebook](tutorials/lightcurve_compare.ipynb) and [Create a phase curve](tutorials/create_a_lightcurve.ipynb)). Since ``squishyplanet`` uses its own implementation of the polynomial limb darkening model presented in [Agol, Luger, and Foreman-Mackey 2020](https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract), it can handle complex limb darkening profiles even while accounting for the planet's non-circular, potentially time-varying, projected shape.

Though it is generally slower than `jaxoplanet` given its increased reliance on numerical solutions and more complex underlying model, `squishyplanet` is also built on `JAX` and can be just-in-time compiled for speed. Users can expect reasonably-sized transit-only calculations to take ~10s of ms. However, phase curve calculations, which rely on Monte Carlo integrations at each timestep, are much slower than transit calculations. So, users should expect phase curve evaluations to be much slower, on the order of 100s of ms per evaluation. Be aware of this when selecting an inference framework if trying to fit actual data. 

Also note that although `JAX` can technically compute the gradient of the likelihood function with respect to the model parameters, as of now, these calculations take *significantly longer* than the forward model evaluations. This is a performance issue that may be revisited in the future, but for now, gradient-dependent techniques like HMC are likely impractical with `squishyplanet`.

We recommend that potential users start with the [geometry visualizations](geometry.rst) to get a sense of the coordinate system and how the planet is defined. Those interested in contributing to the code, or who find issues/want some clarification, should check out the [contributing](contributing.md) page and open an issue or pull request. Happy squishing!

## Attribution

\[insert JOSS citation/bibtex here someday\]




```{toctree}
:maxdepth: 1
:hidden:
:caption: User Guide

installation
quickstart
geometry
contributing
changelog
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Tutorials/Demos

tutorials/illustrations.ipynb
tutorials/lightcurve_compare.ipynb
tutorials/create_a_lightcurve.ipynb
tutorials/create_a_phase_curve.ipynb

```


```{toctree}
:maxdepth: 1
:hidden:
:caption: API

api
```
