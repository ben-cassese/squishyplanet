Geometry visualizations
========================

All of ``squishyplanet``'s calculations rely on a geometric description of a planet in 3D space. All of the translatations/rotations can be tough to visualize, so we try to walk through them here.

We begin in a cartesian coordinate system centered on the planet, oriented so that its north pole falls along the :math:`z`-axis. The planet starts as a sphere with radius $r$, then we compress the sphere along two axes to create a triaxial sphereoid.

The $f_1$ flattening term controls the difference between polar and equatorial radii. You might expect a non-zero $f_1$ for a rapidly spinning planet that bulges out in the middle due to centrifugal forces. Jupiter and Saturn have measured $f_1$ values of
`0.06 <https://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html>`_  and `0.098 <https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html>`_, respectively.

The $f_2$ flattening term controls the difference between the two equatorial radii. You might expect a non-zero $f_2$ for a planet that's very close to its host star and is deformed by tidal forces.

.. video:: _static/media/videos/_static/480p15/sections/SquishyPlanet_0000.mp4
   :loop:
   :autoplay:

After we compress the planet, we orient it with respect to its own orbital plane. First we give is some obliquity via a rotation around the $y$ axis, then we set its precession angle by rotating around the $z$ axis. These values are referred to as ``obliq`` and ``prec`` in the code. 

.. video:: _static/media/videos/_static/480p15/sections/SquishyPlanet_0001.mp4
   :loop:
   :autoplay:

Next we translate the planet to the correct phase-dependent location within its orbital frame. These are set by the usual translations:

.. math::
    \begin{align*}
    r_p &= a + \frac{1-e^2}{1+e \cos(f)}, \\
    x &= r_p \cos(f), \\
    y &= r_p \sin(f), \\
    z &= 0
    \end{align*}

.. video:: _static/media/videos/_static/480p15/sections/SquishyPlanet_0002.mp4
   :loop:
   :autoplay:

where $a$ is the semi-major axis, $e$ is the eccentricity, and $f$ is the true anomaly. This re-centers the coordinate system on the star. Note that these are *translations*,
not *rotations*, so the planet's obliquity-precession orientation does not change as a
function of phase unless we force it to by continuously updating ``prec``. ``squishyplanet`` does this automatically whenever ``tidally_locked`` is set to ``True`` when creating an :ref:`OblateSystem` object.

Finally, we rotate this orbital frame to account for our viewing geometry. We re-define the coordinate system such that we watch the scene unfold from $z = \\infty$ and $x$ is to the right. 

.. video:: _static/media/videos/_static/480p15/sections/SquishyPlanet_0003.mp4
   :loop:
   :autoplay:

After this sequence of rotations/translations, the planet can still be described as a quadratic surface. In it's own frame, the planet can be defined implicitly as:

.. math::
    \frac{x^2}{r^2} + \frac{y^2}{r^2(1-f_2)^2} + \frac{z^2}{r^2(1-f_1)^2} = 1

In our frame, we now have a bunch of somewhat horrendous cross terms, but it's still a quadratic:

.. math::
    p_{xx} x^2 + p_{xy} x y + p_{xz} x z + p_{yy} y^2 + p_{yz} y z + p_{zz} z^2 + p_{x0} x + p_{y0} y + p_{z0} z + p_{00} = 1


The purpose of `squishyplanet.engine.planet_3d.planet_coeffs_3d <https://squishyplanet.readthedocs.io/en/latest/engine.html#planet_3d.planet_3d_coeffs>`_ is to compute these $p$ coefficients from the orbital elements.

For calculations involving the reflected or emitted light of the phase curve, we'll need the normal vector to the surface at each point. Keeping the planet in this implicit form is helpful there, since the normal vector is just the gradient of the implicit function.

For calculations involving the transits however, we don't care about the 3D representation of the planet: we're only senstive to the projected 2D outline. To compute this, we solve for $z$ as a function of $x$ and $y$ where the surface normal of the planet is perpendicular to our line of sight (we can't see "over the horizon" of the planet), then plug that into the 3D implicit equation. That leaves us with a 2D implicit equation for the planet's outline:

.. math::
    \rho_{xx} x^2 + \rho_{xy} x y + \rho_{yy} y^2 + \rho_{x0} x + \rho_{y0} y + \rho_{00} = 1

.. video:: _static/media/videos/_static/480p15/TransitSetup.mp4
   :loop:
   :autoplay:

The purpose of `squishyplanet.engine.planet_2d.planet_coeffs_2d <https://squishyplanet.readthedocs.io/en/latest/engine.html#planet_2d.planet_2d_coeffs>`_ is to compute these $\\rho$ coefficients from the $p$ coefficients.

This is helpful, but still not the most convenient form for further calculations. We will compute the time-dependent flux blocked by the planet using part the algorithm in `Agol, Luger, and Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_. This requires tracing out the boundary of the flux-blocking area and applying Green's theorem to compute the enclosed flux. When the planet overlaps the limb of the star, the portion bounded by the stellar edge is easy to parameterize: it's just a circle. The portion bounded by the planet's edge is more complicated though, so we recast the implicit equation in a parametric form:

.. math::
    \begin{align*}
    x(\alpha) &= c_{x1} \cos(\alpha) + c_{x2} \sin(\alpha) + c_{x3}, \\
    y(\alpha) &= c_{y1} \cos(\alpha) + c_{y2} \sin(\alpha) + c_{y3}
    \end{align*}


The purpose of `squishyplanet.engine.parametric_ellipse.poly_to_parametric <https://squishyplanet.readthedocs.io/en/latest/engine.html#parametric_ellipse.poly_to_parametric?>`_ is to convert between the $\\rho$ and $c$ coefficients.

At each timestep, the workflow is then the following:

1. Solve Kepler's equation for the true anomaly.
2. Compute the $p$ coefficients from the orbital elements.
3. Compute the $\\rho$ coefficients from the $p$ coefficients.
4. Solve for intersections between the planet and the star. This involves finding the roots of a quartic polynomial, which we do numerically.
5. If there are real intersections, or if the planet is completely inside the star, compute the $c$ coefficients of the parametric ellipse. 
6. If there are real intersections

    a. For each intersection point, compute the corresponding $\\alpha$ value.
    b. Marching around the edge of the planet between $\\alpha_1$ and $\\alpha_2$, numerically integrate the flux encountered in the "Green's Basis" `Agol, Luger, and Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_.
    c. Marching around the edge of the star in the same direction, numerically integrate the flux encountered in the Green's Basis.
    d. Add the paths together to form a closed curve and compute the total enclosed flux.

7. If the planet is fully in transit

    a. Marching fully around the edge of the planet from $\\alpha = 0$ to $\\alpha = 2\\pi$, numerically integrate the flux encountered in the Green's Basis.

.. video:: _static/media/videos/_static/480p15/Transit.mp4
   :loop:
   :autoplay:

Note that in this animation, the planet's sky-projected ellipse does not change size or orientation as a function of phase. As mentioned above, that's because this planet is not tidally locked. In this case, it's overkill to use the full 3D parameterization of the planet, since there's an infinite number of flattening/rotation combinations that will get you this same 2D ellipse. This is why when dealing with non-locked planets, users have the option to set ``parameterize_with_projected_elipse`` to ``True`` when creating a :ref:`OblateSystem` object. With this flag enabled, instead of supplying values like ``r``, ``f1``, ``f2``, ``obliq``, and ``prec``, you can just supply ``projected_r``, ``projected_f``, and ``projected_theta``. See `Create a transit lightcurve <https://squishyplanet.readthedocs.io/en/latest/tutorials/create_a_lightcurve.html>`_ for more.

However, if we're dealing with a tidally locked planet, ``squishyplanet`` will keep track of how the planet's projected outline changes with phase. Unlike the non-tidally locked case where the difference between oblate and spherical planets shows up almost entirely during ingress and egress, the time-varying area of a tidally locked planet's projected ellipse can cause significant differences in the light curve at all transit phases. See how the shape of the planet's projected ellipse changes as a function of phase in the video below, and again `Create a transit lightcurve <https://squishyplanet.readthedocs.io/en/latest/tutorials/create_a_lightcurve.html>`_ for more.

.. video:: _static/media/videos/_static/480p15/TidalLocking.mp4
   :loop:
   :autoplay: