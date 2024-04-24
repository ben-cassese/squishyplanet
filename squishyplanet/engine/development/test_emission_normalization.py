# still kind of confused by this: when applying this correction, the total emission
# looks correct but the spatial distribution gets weird/isn't a single hotspot anymore.
# leaving this off for now.

# # spent a while confused about how to normalize the total emission from the planet, so
# # this is a sanity check. When randomly sampling points on the surface of the planet,
# # the average emission should be 1.0, regardless of the shape of the planet or the
# # location/concentration of the hotspot. To make that happen, we have to weight points
# # based on the biases you'd get from uneven densities on the unit sphere after
# # un-squishing the planet. This test is admittedly a little circular, since when
# # generating uniform samples on the planet's surface we use a rejection sampling scheme
# # based on the same factor we'll then use to re-weight the points, but still, it helped
# # to write it out.

# # also note that this is all in the 3D geometry- when we sample/view the planet
# # projected onto the sky frame, need to divide by another factor of 4 since the
# # weighted, un-squished points can be equivalently thought of as samples of the unit
# # disk.

# import jax

# jax.config.update("jax_enable_x64", True)
# import jax.numpy as jnp

# from squishyplanet.engine.phase_curve_utils import (
#     _emission_profile,
#     emission_squish_correction,
# )

# from tqdm import tqdm


# nsamples = 1_000_000


# def sample_ellipsoid(key, n_samples, r, f1, f2):
#     _, *keys = jax.random.split(key, 4)
#     a = r
#     b = r * (1 - f2)
#     c = r * (1 - f1)

#     # sample points uniformly on the sphere
#     s = n_samples * 10
#     theta = jnp.arccos(jax.random.uniform(keys[0], (s,), minval=-1, maxval=1))
#     phi = jax.random.uniform(keys[1], (s,), minval=0, maxval=2 * jnp.pi)

#     x = 1.0 * jnp.sin(theta) * jnp.cos(phi)
#     y = 1.0 * jnp.sin(theta) * jnp.sin(phi)
#     z = 1.0 * jnp.cos(theta)

#     # project points to the ellipsoid
#     x = a * x
#     y = b * y
#     z = c * z

#     # compute rejection probability
#     w = jnp.min(jnp.array([a, b, c])) * jnp.sqrt(
#         x**2 / a**4 + y**2 / b**4 + z**2 / c**4
#     )
#     selected = jax.random.choice(
#         keys[2], jnp.arange(s), p=w / jnp.sum(w), shape=(n_samples,)
#     )
#     x = x[selected]
#     y = y[selected]
#     z = z[selected]

#     return x, y, z


# def test_normalization():
#     planets = []
#     for k in tqdm(range(100)):
#         _, *keys = jax.random.split(jax.random.PRNGKey(k), 8)
#         r = jax.random.uniform(keys[0], minval=0.5, maxval=1.5)
#         f1 = jax.random.uniform(keys[1], minval=0.0, maxval=0.99)
#         f2 = jax.random.uniform(keys[2], minval=0.0, maxval=0.99)
#         hotspot_lat = jax.random.uniform(keys[3], minval=0, maxval=jnp.pi)
#         hotspot_lon = jax.random.uniform(keys[4], minval=0, maxval=2 * jnp.pi)
#         hotspot_concentration = jax.random.uniform(keys[5], minval=0.1, maxval=5.0)

#         planet_x, planet_y, planet_z = sample_ellipsoid(
#             key=keys[6],
#             n_samples=nsamples,
#             r=r,
#             f1=f1,
#             f2=f2,
#         )

#         assert jnp.allclose(
#             planet_x**2 / (r**2)
#             + planet_y**2 / (r**2 * (1 - f2) ** 2)
#             + planet_z**2 / (r**2 * (1 - f1) ** 2),
#             1.0,
#         )

#         unnormalized_planet = _emission_profile(
#             x=planet_x,
#             y=planet_y,
#             z=planet_z,
#             r=r,
#             f1=f1,
#             f2=f2,
#             hotspot_latitude=hotspot_lat,
#             hotspot_longitude=hotspot_lon,
#             hotspot_concentration=hotspot_concentration,
#         )

#         planet_correction = emission_squish_correction(
#             x=planet_x,
#             y=planet_y,
#             z=planet_z,
#             r=r,
#             f1=f1,
#             f2=f2,
#         )

#         planet = jnp.sum(unnormalized_planet * planet_correction) / jnp.sum(
#             planet_correction
#         )

#         planets.append(planet)

#     planets = jnp.array(planets)

#     # the planet samples are all close to 1:
#     assert jnp.abs(jnp.mean(planets) - 1) < 0.01

#     # and that the distribution is close to normal:
#     s = jnp.std(planets)
#     assert jnp.abs(jnp.mean(planets) - 1) < 3 * s


# test_normalization()
