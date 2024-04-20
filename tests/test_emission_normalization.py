# # I got confused about the correction needed when using the von Mises-Fisher
# # distribution but on the surface of an ellipsoid instead of a sphere.

# # but, think this is unnecessary now

# import jax
# jax.config.update("jax_enable_x64", True)
# import jax.numpy as jnp

# from squishyplanet.engine.phase_curve_utils import _emission_profle

# from scipy.stats import kstest
# from tqdm import tqdm


# def test_normalization():

#     sphere_values = []
#     planet_values = []

#     for i in tqdm(range(3)):
#         n = 1_000_000
#         key = jax.random.PRNGKey(i)
#         _, *keys = jax.random.split(key, num=13)
#         r = jax.random.uniform(keys[0], (), minval=0.01, maxval=0.5)
#         f1 = jax.random.uniform(keys[1], (), minval=0.001, maxval=0.5)
#         _f2 = jax.random.uniform(keys[2], (), minval=0.0, maxval=1)
#         f2 = _f2*f1
#         hotspot_latitude = jax.random.uniform(keys[3], ()) * jnp.pi
#         hotspot_longitude = jax.random.uniform(keys[4], ()) * 2 * jnp.pi
#         hotspot_concentration = jax.random.uniform(keys[5], (), minval=0.01, maxval=10)


#         # samples of the planet:
#         theta = jnp.arccos(jax.random.uniform(keys[6], (n,), minval=-1, maxval=1))
#         phi = jax.random.uniform(keys[7], (n,)) * 2 * jnp.pi
#         planet = jnp.array(
#             [
#                 r * jnp.cos(phi) * jnp.sin(theta),
#                 r * jnp.sin(phi) * jnp.sin(theta),
#                 r * jnp.cos(theta),
#             ]
#         ).T
#         planet = planet.at[:, 2].set(planet[:, 2] * jnp.sqrt((1 - f1) ** 2))
#         planet = planet.at[:, 1].set(planet[:, 1] * jnp.sqrt((1 - f2) ** 2))
#         # at this point, we should have biased samples on the planet surface
#         # concentrated at the x poles


#         # deformation = emission_squish_correction(
#         #     x=planet[:,0],
#         #     y=planet[:,1],
#         #     z=planet[:,2],
#         #     r=r,
#         #     f1=f1,
#         #     f2=f2)/r

#         # rands = jax.random.uniform(keys[8], (n,), minval=0, maxval=1)
#         # selected = rands < deformation
#         # planet = planet[selected]
#         # weights = deformation[selected]

#         assert jnp.allclose(
#             planet[:, 0] ** 2 / (r**2)
#             + planet[:, 1] ** 2 / (r**2 * (1 - f2) ** 2)
#             + planet[:, 2] ** 2 / (r**2 * (1 - f1) ** 2),
#             1.0,
#         )

#         # samples of a perfect sphere:
#         theta = jnp.arccos(jax.random.uniform(keys[9], (len(planet),), minval=-1, maxval=1))
#         phi = jax.random.uniform(keys[10], (len(planet),)) * 2 * jnp.pi
#         sphere = jnp.array(
#             [
#                 r * jnp.cos(phi) * jnp.sin(theta),
#                 r * jnp.sin(phi) * jnp.sin(theta),
#                 r * jnp.cos(theta),
#             ]
#         ).T
#         assert jnp.allclose(
#             sphere[:, 0] ** 2 / (r**2)
#             + sphere[:, 1] ** 2 / (r**2)
#             + sphere[:, 2] ** 2 / (r**2),
#             1.0,
#         )


#         sphere_sum = jnp.sum(
#             _emission_profle(
#                 x=sphere[:, 0],
#                 y=sphere[:, 1],
#                 z=sphere[:, 2],
#                 r=r,
#                 f1=0.0,
#                 f2=0.0,
#                 hotspot_latitude=hotspot_latitude,
#                 hotspot_longitude=hotspot_longitude,
#                 hotspot_concentration=hotspot_concentration,
#             )
#         )

#         planet_sum = jnp.sum(
#             _emission_profle(
#                 x=planet[:, 0],
#                 y=planet[:, 1],
#                 z=planet[:, 2],
#                 r=r,
#                 f1=f1,
#                 f2=f2,
#                 hotspot_latitude=hotspot_latitude,
#                 hotspot_longitude=hotspot_longitude,
#                 hotspot_concentration=hotspot_concentration,
#             )
#         )

#         sphere_values.append(sphere_sum/len(planet))
#         planet_values.append(planet_sum/len(planet))

#     sphere_values = jnp.array(sphere_values)
#     planet_values = jnp.array(planet_values)

#     # the sphere samples gave the expected result
#     assert jnp.abs(jnp.mean(sphere_values) - 1) < 0.01

#     # the planet samples gave the expected result
#     assert jnp.abs(jnp.mean(planet_values) - 1) < 0.01

#     # the distributions look similar
#     assert kstest(planet_values, sphere_values).pvalue > 0.05

#     m = jnp.mean(planet_values - sphere_values)
#     std = jnp.std(planet_values - sphere_values)

#     # the signed difference is consistent with zero
#     assert m / std < 2


# # @jax.jit
# # def emission_squish_correction(x,y,z,r,f1,f2):
# #     """
# #     Correction factor for the squishing of the planet due to its oblateness.

# #     We're using the
# #     `von Mises-Fisher distribution
# #     <https://https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution>`_ to
# #     model a hotspot. But, that's defined on the unit sphere, and after compressing it to
# #     the squished planet, the surface denisty of emission intensity will be warped. We
# #     need to correct for that warping here. The input coordinates here are **IN THE
# #     PLANET'S FRAME, NOT THE SKY FRAME.** After getting :math:`x,y,z` samples in the sky
# #     frame, apply the rotation matrix from :func:`pre_squish_transform` to get these.

# #     Not sure if this link will live, but see
# #     `here <https://math.stackexchange.com/questions/973101/how-to-generate-points-uniformly-distributed-on-the-surface-of-an-ellipsoid>`_
# #     for an intuition.

# #     Args:
# #         x (Array):
# #             The x values of the points on the planet's surface IN THE PLANET'S FRAME
# #         y (Array):
# #             The y values of the points on the planet's surface IN THE PLANET'S FRAME
# #         z (Array):
# #             The z values of the points on the planet's surface IN THE PLANET'S FRAME
# #         r (Array):
# #             The equatorial radius of the planet.
# #         f1 (Array):
# #             The planet's :math:`z` flattening coefficient.
# #         f2 (Array):
# #             The planet's :math:`y` flattening coefficient.

# #     Returns:
# #         Array:
# #             The correction factor for the squishing of the planet due to its oblateness.


# #     """
# #     a = 1/r
# #     b = 1/jnp.sqrt(r**2 * (1 - f2)**2)
# #     c = 1/jnp.sqrt(r**2 * (1 - f1)**2)

# #     # this will be less than one away from the poles
# #     area_after_squish = (jnp.sqrt(
# #         (a*c*y)**2 + (a*b*z)**2 + (b*c*x)**2
# #     ) / (b*c))

# #     return area_after_squish
