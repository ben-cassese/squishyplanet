---
title: '`squishyplanet`: modeling transits of non-spherical exoplanets in JAX'
tags:
  - Python
  - astronomy
  - exoplanets
  - exoplanet transits
authors:
  - name: Ben Cassese
    orcid: 0000-0002-9544-0118
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Justin Vega
    orcid: 0000-0003-1481-8076
    affiliation: 1
  - name: Tiger Lu
    orcid: 0000-0003-0834-8645
    affiliation: 2
  - name: Malena Rice
    orcid: 0000-0002-7670-670X
    affiliation: 2
  - name: David Kipping
    orcid: 0000-0002-4365-7366
    affliation: 1
affiliations:
 - name: Dept. of Astronomy, Columbia University, 550 W 120th Street, New York NY 10027, USA
   index: 1
 - name: Dept. of Astronomy, Yale University, New Haven, CT 06511, USA
   index: 2
date: 01 May 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Although the majority of exoplanets are expected to be nearly spherical, a subset is expected to be distinctly oblate or even triaxial. These include both gas giants that are close to their host stars and are consequently stretched by tidal forces, and those that orbit further out but likely rotate rapidly due to an inability to shed a large angular momentum budget inherited from their formation. These shape deformations will imprint themselves on the resulting transit lightcurve, especially during ingress and egress as the planet partially overlaps with the limb of the star. Correctly modeling/fitting for the parameters that describe deviations from a perfect sphere could enable new insights into planetary structure and eliminate the biases that result from fitting a mis-specified spherical model. We present `squishyplanet`, a JAX-based Python package that implements an extension of the polynomial limb-darkened transit model presented in @alfm to non-spherical (triaxial) planets, as well as routines for modeling reflection and emission phase curves.


# Statement of need

The study of exoplanets, or planets that orbit stars beyond the sun, is a major focus of the astronomy community. One of the most common and powerful types of data used in this process are timeseries photometric (or spectroscopic) observations collected when a planet happens to pass through the line of sight between an observer and its host star. By modeling the fraction of starlight intercepted by the coincident planet, astronomers can deduce basic properties of the system such as the planet's relative size, its orbital period, and its orbital inclination.

The past 20 years have seen extensive work both on theoretical model development and computationally efficient implementations of these models. Notable examples include @mandel_agol, @batman, and @exoplanet, though many other examples can be found. Though each of these packages make different choices, the majority of them (with notable exceptions, including @ellc) do share one common assumption: the planet under examination is a perfect sphere.

This is both a reasonable and immensely practical assumption. It is reasonable since firstly, a substantial fraction of planets, especially rocky planets, are likely quite close to perfect spheres (Earth's equatorial radius is only 43 km greater than its polar radius, a difference of only 0.3%). Secondly, at the precision of most survey datasets (e.g. *Kepler* and *TESS*), even substantially flattened planets would be nearly indistinguishable from a spherical planet with the same on-sky projected area. It is practical since, somewhat miraculously, this assumption enables an analytic solution to the amount of flux blocked by the planet at each timestep. This is true even if the intensity of the stellar surface varies radially according to a nearly arbitrarily complex polynomial [@alfm].

However, for a small but growing number of datasets and targets, the reasonableness of this assumption will break down and lead to biased results. Many gas giant planets in particular are expected to be distinctly oblate or triaxial, either due to the effects of tidal deformation or rapid rotation. Looking within our own solar system, Jupiter and Saturn have oblateness values of roughly 0.06 and 0.1, respectively, due to their fast spins.

To illustrate the effects of shape deformation on a lightcurve, consider \autoref{fig:example}, which shows a selection of differences between timeseries generated under the assumption of a spherical planet and those generated assuming a planet with Saturn-like flattening. Depending on the obliquity, precession, impact parameter, and whether the planet is tidally locked, we can generate a wide variety of residual lightcurves. In some cases the deviations from a spherical planet occur almost exclusively in the ingress and egress phases of the transit, while others evolve throughout the transit. Some residual curves are symmetric about the transit midpoint, though in general, they will not be.

![A sampling differences between spherical and non-spherical planets during their transits. A fuller description of how each of these curves were generated can be found in the [online documentation](https://squishyplanet.readthedocs.io/en/latest/tutorials/create_a_lightcurve.html).\label{fig:example}](deviations.png)

The amplitude of these effects are quite small compared to the full depth of the transit, but could be easily detectable with a facility such as JWST, which now routinely achieves a white-light precision of a few 10s of ppm [@ERS_prism].

We leave a detailed description of the mathematics and a corresponding series of visualizations for the online documentation. There we also include confirmation that our implementation, when modeling the limiting case of a spherical planet, agrees with previous well-tested models even for high-order polynomial limb darkening laws. More specifically, we show that that lightcurves of spherical planets generated with `squishyplanet` deviate by no more than 100 ppb from those generated with  `jaxoplanet` [@jaxoplanet], the JAX-based rewrite of the popular transit modeling package `exoplanet` [@exoplanet] that also implements the arbitrary-order polynomial limb darkening algorithm presented in @alfm. Finally, we demonstrate `squishyplanet`'s limited support for phase curve modeling. 

We hope that a publicly-available, well-documented, and highly accurate model for non-spherical exoplanet transits will enable thorough studies of planets' shapes and lead to more data-grounded constraints on their interior structures.

# Acknowledgements

We acknowledge many helpful discussions with Avishi Poddar at the early stages of this project. ``squishyplanet`` also relies on ``quadax`` [@quadax], an open-source library for numerical quadrature and integration using JAX.

# References