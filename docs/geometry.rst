Geometry visualizations
========================

All of ``squishyplanet``'s calculations rely on a geometric description of a planet in 3D space. These translatations/rotations can be tough to visualize, so here we try to break down the underlying process step by step.

3D Orientation
^^^^^^^^^^^^^^

We begin in a cartesian coordinate system centered on the planet, oriented so that its north pole falls along the :math:`z`-axis. The planet starts out as a sphere with radius $r$, then is flattened into a triaxial sphereoid via compression along two axes.

The $f_1$ flattening term controls the difference between polar and equatorial radii. You might expect a non-zero $f_1$ for a rapidly spinning planet that bulges out in the middle due to centrifugal forces. Jupiter and Saturn have measured $f_1$ values of
`0.06 <https://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html>`_ and `0.098 <https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html>`_, respectively.

The $f_2$ flattening term controls the difference between the two equatorial radii. You might expect a non-zero $f_2$ for a planet that's very close to its host star and is deformed by tidal forces. In our coordinate system, the $y$-axis is the one that gets squished by $f_2$, meaning that after these compressions, the planet still extends out to $x=r$, while its maximum $z$ extent is $r(1-f_1)$ and its maximum $y$ extent is $r(1-f_2)$.

.. video:: _static/media/videos/_static/480p15/sections/SquishyPlanet_0000.mp4
   :loop:
   :autoplay:

After we compress the planet, we orient it with respect to its own orbital plane. First we give it some obliquity via a rotation around the $y$ axis, then we set its precession angle by rotating around the $z$ axis. These values are referred to as ``obliq`` and ``prec`` in :ref:`OblateSystem` objects.

.. video:: _static/media/videos/_static/480p15/sections/SquishyPlanet_0001.mp4
   :loop:
   :autoplay:

Next we translate the planet to the correct phase-dependent location within its orbital plane. These are set by the usual transformations:

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

where $a$ is the semi-major axis, $e$ is the eccentricity, and $f$ is the true anomaly. These re-center the coordinate system on the star. Note that these are *translations*,
not *rotations*, so the planet's obliquity-precession orientation does not change as a
function of phase unless we force it to by continuously updating ``prec``. ``squishyplanet`` does this automatically whenever ``tidally_locked`` is set to ``True`` when creating an :ref:`OblateSystem` object.

Finally, we rotate this orbital frame to account for our viewing geometry. We re-define the coordinate system such that we watch the scene unfold from $z = \\infty$ and $x$ is to the right.

.. video:: _static/media/videos/_static/480p15/sections/SquishyPlanet_0003.mp4
   :loop:
   :autoplay:

These rotations *do* alter the projected orientation of the planet. After this sequence of rotations/translations, the planet can still be described as a quadratic surface. In its own original frame, the planet can be defined implicitly as:

.. math::
    \frac{x^2}{r^2} + \frac{y^2}{r^2(1-f_2)^2} + \frac{z^2}{r^2(1-f_1)^2} = 1

In our frame, we now have a bunch of somewhat horrendous cross terms, but it's still a quadratic:

.. math::
    p_{xx} x^2 + p_{xy} x y + p_{xz} x z + p_{yy} y^2 + p_{yz} y z + p_{zz} z^2 + p_{x0} x + p_{y0} y + p_{z0} z + p_{00} = 1


The purpose of `squishyplanet.engine.planet_3d.planet_coeffs_3d <https://squishyplanet.readthedocs.io/en/latest/engine.html#planet_3d.planet_3d_coeffs>`_ is to compute these $p$ coefficients from the orbital elements at each timestep.

2D Projection
^^^^^^^^^^^^^

For calculations involving the reflected or emitted light of the phase curve, we'll need the normal vector to the surface at each point. Keeping the planet in this implicit form is helpful there, since the normal vector is just the gradient of the implicit function.

For calculations involving the transits however, we don't care about the 3D representation of the planet: we're only sensitive to the projected 2D outline. To compute this, we solve for $z$ as a function of $x$ and $y$ where the surface normal of the planet is perpendicular to our line of sight (we can't see "over the horizon" of the planet), then plug that into the 3D implicit equation. That leaves us with a 2D implicit equation for the planet's outline:

.. math::
    \rho_{xx} x^2 + \rho_{xy} x y + \rho_{yy} y^2 + \rho_{x0} x + \rho_{y0} y + \rho_{00} = 1

.. video:: _static/media/videos/_static/480p15/TransitSetup.mp4
   :loop:
   :autoplay:

The purpose of `squishyplanet.engine.planet_2d.planet_coeffs_2d <https://squishyplanet.readthedocs.io/en/latest/engine.html#planet_2d.planet_2d_coeffs>`_ is to compute these $\\rho$ coefficients from the $p$ coefficients.

Parametric Form
^^^^^^^^^^^^^^^

This is helpful, but still not the most convenient form for further calculations. We will compute the time-dependent flux blocked by the planet using part the algorithm in `Agol, Luger, and Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_. This requires tracing out the boundary of the flux-blocking area and applying Green's theorem to compute the enclosed flux. When the planet overlaps the limb of the star, the portion bounded by the stellar edge is easy to parameterize: it's just a circle. The portion bounded by the planet's edge is more complicated though, so we recast the implicit 2D equation in a parametric form:

.. math::
    \begin{align*}
    x(\alpha) &= c_{x1} \cos(\alpha) + c_{x2} \sin(\alpha) + c_{x3}, \\
    y(\alpha) &= c_{y1} \cos(\alpha) + c_{y2} \sin(\alpha) + c_{y3}
    \end{align*}

The purpose of `squishyplanet.engine.parametric_ellipse.poly_to_parametric <https://squishyplanet.readthedocs.io/en/latest/engine.html#parametric_ellipse.poly_to_parametric?>`_ is to convert between the $\\rho$ and $c$ coefficients.

We can now fully describe the boundary of the flux-blocking area as a 1D parametric curve: it'll be a piecewise curve when the planet overlaps with the limb of the star, and just the outline of the planet itself when it's fully in transit. This description lets us numerically integrate the equations that are largely solved analytically in `Agol, Luger, and Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_, which describe the flux blocked by 2D shape overlapping a portion of a stellar disk described by a polynomial limb darkening law.

.. video:: _static/media/videos/_static/480p15/Transit.mp4
   :loop:
   :autoplay:


Green's Basis
^^^^^^^^^^^^^

We happily leave a detailed explanation of this process to Agol et al., who provide an excellent description of the algorithm in the paper above (which includes links to code examples and complete derivations). However, when adapting the algorithm for this more general case we have made two additions that are worth documenting.

We assume the star's intensity profile is described as:

.. math::

    \frac{I(z)}{I_0} = 1 - u_1(1-z) - u_2(1-z)^2 - ... - u_n(1-z)^n = \tilde{u}^T \vec{u}

where $\\tilde{u}$ is the "limb darkening basis" and $\\vec{u}$ is the vector of limb darkening coefficients (Eq. 3). We then find the change of basis matrix $A$ that converts $\\vec{u}$ into $\\vec{g}$, a transformed set of coeffients, that we will multiply with the "Green's basis", $\\tilde{g}$. This new basis takes the form:

.. math::

    \tilde{g}_{n}=\begin{cases}
    1&n=0\\ z&n=1\\ (n+2)z^{n}-nz^{n-2}&n\ge2\end{cases}

in Eq. 14. Our total flux is now $\\tilde{u}^T \\vec{u} = \\tilde{g}^T A \\vec{u} =  \\tilde{g}^T \\vec{g}$, where $\\vec{g}$ is our vector of transformed $u$ coefficients. This somewhat odd-looking basis is chosen because it enables a surprisingly elegant form for applying Green's theorem to compute the blocked flux. If we also define $G_{n}(z) = z^n (-y \\hat{x} + x \\hat{y})$ (Eq. 62), note that:

.. math::

    \frac{dG_{n,y}}{dx} - \frac{dG_{n,x}}{dy} = (n+2)z^{n}-nz^{n-2} = \tilde{g}_{n}

Now we can use a 1D integral of $G_n(\\mu)$ dotted with a closed path $dr$ to compute the flux enclosed by that path:

.. math::

     \int \int I(x,y) dA = \int \int \left( \frac{dG_{n,y}}{dx} - \frac{dG_{n,x}}{dy} \right) dx dy = \oint G_n(z) \cdot dr

Where $\dr$ is $\\{dx, dy\\} = \\{dx(\\alpha), dy(\\alpha)\\}$ from the parametric form of the ellipse above.

Summing these integrals over all $n$ gives the total flux enclosed by the path. But, the above works only for $n\\geq2$, and Agol, Luger, and Foreman-Mackey 2020 do not explicitly provide $G_0$ and $G_1$. They don't need to, since for a spherical planet they can skip straight to analytic solutions for these low-order laws. Indeed, they go on to derive analytic solutions even for these higher-order terms, and only report the explicit form of $G_n$ as an intermediate step. However, since our planets aren't spherical, their projected outlines are no longer simple circles, so we need to compute these integrals numerically even for the lowest-order terms.

We use the following forms for $G_0$ and $G_1$:

.. math::

    G_0 = \{0, x\}

.. math::

    G_1 = \left\{0, \frac{1}{2} \left(x \sqrt{-x^2-y^2+1}-\left(y^2-1\right) \tan ^{-1}\left(\frac{x}{\sqrt{-x^2-y^2+1}}\right)\right)+\frac{\pi }{12} \right\}

We use an implementation of Gauss-Konrod quadrature in the open source `quadax <https://github.com/f0uriest/quadax/tree/main>`_ package for these integrals. For more, see the `api documentation <https://https://squishyplanet.readthedocs.io/en/latest/engine.html#polynomial_limb_darkened_transit.planet_solution_vec>`_.

Putting It Together
^^^^^^^^^^^^^^^^^^^

At each timestep, the workflow is then the following:

1. Solve Kepler's equation for the true anomaly.
2. Compute the $p$ coefficients from the orbital elements.
3. Compute the $\\rho$ coefficients from the $p$ coefficients.
4. Solve for intersections between the planet and the star. This involves finding the roots of a quartic polynomial, which we do numerically.
5. If there are real intersections, or if the planet is completely inside the star, compute the $c$ coefficients of the parametric ellipse.
6. If there are real intersections

    a. For each intersection point, compute the corresponding $\\alpha$ value.
    b. Marching around the edge of the planet between $\\alpha_1$ and $\\alpha_2$, numerically integrate the flux encountered in the "Green's Basis" (`Agol, Luger, and Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_).
    c. Marching around the edge of the star in the same direction, numerically integrate the flux encountered in the Green's Basis.
    d. Add the paths together to form a closed curve and compute the total enclosed flux.

7. If the planet is fully in transit

    a. Marching fully around the edge of the planet from $\\alpha = 0$ to $\\alpha = 2\\pi$, numerically integrate the flux encountered in the Green's Basis.

Note that `squishyplanet` does not actually scan through these steps at each time step as written since certain steps can be efficiently vectorized across all times at once. But this is essentially what's happening under the hood.

Note on Tidal Locking/Changing Projected Area
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that in the above animations, the planet's sky-projected ellipse does not change size or orientation as a function of phase. As mentioned above, that's because this planet is not tidally locked: changing its true anomaly just translates it within its orbital plane, it does not automatically rotate it as well. In this case, it's overkill to use the full 3D parameterization of the planet, since there's an infinite number of flattening/rotation combinations that will get you this same 2D ellipse. This is why when dealing with non-locked planets, users have the option to set ``parameterize_with_projected_elipse`` to ``True`` when creating a :ref:`OblateSystem` object. With this flag enabled, instead of supplying values like ``r``, ``f1``, ``f2``, ``obliq``, and ``prec``, you can just supply ``projected_r``, ``projected_f``, and ``projected_theta``. See `Create a transit lightcurve <https://squishyplanet.readthedocs.io/en/latest/tutorials/create_a_lightcurve.html>`_ for more.

However, if we're dealing with a tidally locked planet, ``squishyplanet`` will keep track of how the planet's projected outline changes with phase. Unlike the non-tidally locked case where the difference between oblate and spherical planets shows up almost entirely during ingress and egress, the time-varying area of a tidally locked planet's projected ellipse can cause significant differences in the lightcurve at all transit phases. See how the shape of the planet's projected ellipse changes as a function of phase in the video below, and again `Create a transit lightcurve <https://squishyplanet.readthedocs.io/en/latest/tutorials/create_a_lightcurve.html>`_ for more.

.. video:: _static/media/videos/_static/480p15/TidalLocking.mp4
   :loop:
   :autoplay:
