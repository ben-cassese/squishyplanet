# Geometry visualizations


We begin in a cartesian coordinate system centered on the planet, oriented so that it's north pole falls along the $z$-axis. The planet starts as a sphere with radius $r$, then we compress the sphere along two axes to create a triaxial sphereoid.

The $f_1$ flattening term controls the difference between polar and equatorial radii. You might expect a non-zero $f_1$ for a rapidly spinning planet that bulges out in the middle due to centrifugal forces. Jupiter and Saturn have measured $f_1$ values of [0.06](https://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html) and [0.098](https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html), respectively.

The $f_2$ flattening term controls the difference between the two equatorial radii. You might expect a non-zero $f_2$ for a planet that's very close to its host star and is deformed by tidal forces.

![](./visualizations/SphereToEllipsoid.gif)

After we compress the planet, we orient it with respect to its own orbital plane. First we give is some obliquity via a rotation around the $y$ axis, then we set its precession angle by rotating around the $z$ axis. These values are referred to as ``obliq`` and ``prec`` in the code. 

![](./visualizations/SphereToEllipsoid.gif)

Next we translate the planet to the correct phase-dependent location within its orbital frame. These are set by the usual translations:

$$
\begin{align*}
r_p &= a + \frac{1-e^2}{1+e \cos(f)}, \\
x &= r_p \cos(f), \\
y &= r_p \sin(f), \\
z &= 0
\end{align*}
$$

where $a$ is the semi-major axis, $e$ is the eccentricity, and $f$ is the true anomaly. This re-centers the coordinate system on the star.

Finally, we rotate this orbital frame to account for our viewing geometry. We re-define the coordinate system such that we watch the scene unfold from $z = \infty$ and $x$ is to the right. 

![](./visualizations/SphereToEllipsoid.gif)

Note, after this sequence of rotations/translations, the planet can still be described as a quadratic surface. In it's own frame, the planet can be defined implicitly as:

$$
\frac{x^2}{r^2} + \frac{y^2}{r^2(1-f_2)^2} + \frac{z^2}{r^2(1-f_1)^2} = 1
$$

In our frame, we now have a bunch of somewhat horrendous cross terms, but it's still a quadratic:

$$
p_{xx} x^2 + p_{xy} x y + p_{xz} x z + p_{yy} y^2 + p_{yz} y z + p_{zz} z^2 + p_{x0} x + p_{y0} y + p_{z0} z + p_{00} = 1
$$


The purpose of [squishyplanet.engine.planet_3d.planet_coeffs_3d](https://squishyplanet.readthedocs.io/en/latest/engine.html#planet_3d.planet_3d_coeffs) is to compute these $p$ coefficients from the orbital elements.

For calculations involving the reflected or emitted light of the phase curve, we'll need the normal vector to the surface at each point. Keeping the planet in this implicit form is helpful there, since the normal vector is just the gradient of the implicit function.

For calculations involving the transits however, we don't care about the 3D representation of the planet: we're only senstive to the projected 2D outline. To compute this, we solve for $z$ as a function of $x$ and $y$ where the surface normal of the planet is perpendicular to our line of sight (we can't see "over the horizon" of the planet), then plug that into the 3D implicit equation. That leaves us with a 2D implicit equation for the planet's outline:

$$
\rho_{xx} x^2 + \rho_{xy} x y + \rho_{yy} y^2 + \rho_{x0} x + \rho_{y0} y + \rho_{00} = 1
$$

![](./visualizations/SphereToEllipsoid.gif)

The purpose of [squishyplanet.engine.planet_2d.planet_coeffs_2d](https://squishyplanet.readthedocs.io/en/latest/engine.html#planet_2d.planet_2d_coeffs) is to compute these $\rho$ coefficients from the $p$ coefficients.

This is helpful, but still not the most convenient form for further calculations. We will compute the time-dependent flux blocked by the planet using part the algorithm in [Agol, Luger, and Foreman-Mackey 2020](https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract). This requires tracing out the boundary of the flux-blocking area and applying Green's theorem to compute the enclosed flux. When the planet overlaps the limb of the star, the portion bounded by the stellar edge is easy to parameterize: it's just a circle. The portion bounded by the planet's edge is more complicated though, so we recast the implicit equation in a parametric form:

$$
\begin{align*}
x(\alpha) &= c_{x1} \cos(\alpha) + c_{x2} \sin(\alpha) + c_{x3}, \\
y(\alpha) &= c_{y1} \cos(\alpha) + c_{y2} \sin(\alpha) + c_{y3}
\end{align*}
$$

![](./visualizations/SphereToEllipsoid.gif)

The purpose of [squishyplanet.engine.parametric_ellipse.poly_to_parametric](https://squishyplanet.readthedocs.io/en/latest/engine.html#parametric_ellipse.poly_to_parametric) is to convert between the $\rho$ and $c$ coefficients.


<!-- ```{figure} ./visualizations/SphereToEllipsoid.gif
---
width: 100%
figclass: margin-caption
alt: My figure text
name: myfig5
---
test margin caption
``` -->
