# rkey = jax.random.PRNGKey(10)
# for i in tqdm(range(1000)):
#     rkey, *subkeys = jax.random.split(rkey, num=11)

#     ugh = False
#     for f in jnp.linspace(0, 2*jnp.pi, 100):
#         f1 = jax.random.uniform(subkeys[5], minval=0.0, maxval=0.8)
#         state = {
#             "a" : jax.random.uniform(subkeys[0], minval=1.1, maxval=300),
#             "e" : jax.random.uniform(subkeys[1], minval=0.0, maxval=0.99),
#             "i" : jax.random.uniform(subkeys[2], minval=0.0, maxval=jnp.pi),
#             "Omega" : jax.random.uniform(subkeys[3], minval=0.0, maxval=2*jnp.pi),
#             "omega" : jax.random.uniform(subkeys[4], minval=0.0, maxval=2*jnp.pi),
#             "f" : f,
#             "f1" : f1,
#             "f2" : jax.random.uniform(subkeys[6], minval=0.0, maxval=f1),
#             "r" : jax.random.uniform(subkeys[7], minval=0.0, maxval=1.0),
#             "phi" : jax.random.uniform(subkeys[8], minval=0.0, maxval=jnp.pi),
#             "theta" : jax.random.uniform(subkeys[9], minval=0.0, maxval=2*jnp.pi),
#         }
#         s = System(state)
#         c = poly_to_parametric(**s._coeffs_2d)
#         for key in c.keys():
#             if jnp.isnan(c[key]):
#                 print(f)
#                 print(c)
#                 ugh = True
#         if ugh: break



# rkey = jax.random.PRNGKey(13)

# for i in range(10):
#     rkey, *subkeys = jax.random.split(rkey, num=12)

#     ugh = False
#     f1 = jax.random.uniform(subkeys[5], minval=0.0, maxval=0.8)
#     r = jax.random.uniform(subkeys[7], minval=0.0, maxval=1.0)
#     state = {
#         "a" : jax.random.uniform(subkeys[0], minval=1.1, maxval=300),
#         "e" : jax.random.uniform(subkeys[1], minval=0.0, maxval=0.99),
#         "i" : jax.random.uniform(subkeys[2], minval=0.0, maxval=jnp.pi),
#         "Omega" : jax.random.uniform(subkeys[3], minval=0.0, maxval=2*jnp.pi),
#         "omega" : jax.random.uniform(subkeys[4], minval=0.0, maxval=2*jnp.pi),
#         "f" : jax.random.uniform(subkeys[10], minval=0.0, maxval=2*jnp.pi),
#         "f1" : f1,
#         "f2" : jax.random.uniform(subkeys[6], minval=0.0, maxval=f1),
#         "r" : r,
#         "phi" : jax.random.uniform(subkeys[8], minval=0.0, maxval=jnp.pi),
#         "theta" : jax.random.uniform(subkeys[9], minval=0.0, maxval=2*jnp.pi),
#     }
#     s = System(state)
#     c = poly_to_parametric(**s._coeffs_2d)

#     fig, ax = plt.subplots()
#     x = jnp.linspace(c["c_x3"] - r, c["c_x3"] + r, 1000)
#     y = jnp.linspace(c["c_y3"] - r, c["c_y3"] + r, 1000)
#     X, Y = jnp.meshgrid(x, y)
#     planet = (
#         s._coeffs_2d["rho_xx"] * X**2
#         + s._coeffs_2d["rho_yy"] * Y**2
#         + s._coeffs_2d["rho_xy"] * X * Y
#         + s._coeffs_2d["rho_x0"] * X
#         + s._coeffs_2d["rho_y0"] * Y
#         + s._coeffs_2d["rho_00"]
#     )

#     ax.contour(X, Y, planet, levels=[1.0], colors="red", linewidths=10)

#     parametric_angle = jnp.linspace(0, 2*jnp.pi, 1000)
#     x_vals = c["c_x1"] * jnp.cos(parametric_angle) + c["c_x2"] * jnp.sin(parametric_angle) + c["c_x3"]
#     y_vals = c["c_y1"] * jnp.cos(parametric_angle) + c["c_y2"] * jnp.sin(parametric_angle) + c["c_y3"]
#     ax.plot(x_vals, y_vals, color="blue", linewidth=1)

#     ax.set(xlim=(c["c_x3"] - r, c["c_x3"] + r), ylim=(c["c_y3"] - r, c["c_y3"] + r), aspect="equal")


# rkey = jax.random.PRNGKey(13)
# for i in tqdm(range(1000)):
#     rkey, *subkeys = jax.random.split(rkey, num=11)

#     ugh = False
#     for f in jnp.linspace(0, 2*jnp.pi, 100):
#         f1 = jax.random.uniform(subkeys[5], minval=0.0, maxval=0.8)
#         state = {
#             "a" : jax.random.uniform(subkeys[0], minval=1.1, maxval=300),
#             "e" : jax.random.uniform(subkeys[1], minval=0.0, maxval=0.99),
#             "i" : jax.random.uniform(subkeys[2], minval=0.0, maxval=jnp.pi),
#             "Omega" : jax.random.uniform(subkeys[3], minval=0.0, maxval=2*jnp.pi),
#             "omega" : jax.random.uniform(subkeys[4], minval=0.0, maxval=2*jnp.pi),
#             "f" : f,
#             "f1" : f1,
#             "f2" : jax.random.uniform(subkeys[6], minval=0.0, maxval=f1),
#             "r" : jax.random.uniform(subkeys[7], minval=1e-4, maxval=1.0),
#             "phi" : jax.random.uniform(subkeys[8], minval=0.0, maxval=jnp.pi),
#             "theta" : jax.random.uniform(subkeys[9], minval=0.0, maxval=2*jnp.pi),
#         }
#         s = System(state)
#         positions = skypos(**state)
#         c = terminator_planet_intersections(**s._coeffs_3d, x_c=positions[0], y_c=positions[1], z_c=positions[2])
#         for i in c:
#             if jnp.sum(jnp.isnan(i)) > 0:
#                 print(state)
#                 print(c)
#                 ugh = True
#             if jnp.sum(jnp.isinf(i)) > 0:
#                 print(state)
#                 print(c)
#                 ugh = True
#             if jnp.sum(jnp.imag(i)) > 0:
#                 print(state)
#                 print(c)
#                 ugh = True
#         if ugh: break
#     if ugh: break

def light_curve_compare(key, poly_limbdark_order, return_lc=False):
    t = jnp.linspace(-1, 1, 17280) * ureg.day # 10s cadence for 48 hours

    key, *rand_key = jax.random.split(key, num=8)

    u = jax.random.uniform(rand_key[6], shape=(poly_limbdark_order,))
    star_mass = jax.random.uniform(rand_key[0],  minval=0.1, maxval=1.5) * ureg.M_sun
    semimajor_axis = jax.random.uniform(rand_key[1], minval=0.005, maxval=5.0) * ureg.au
    impact_param = jax.random.uniform(rand_key[2], minval=0.0, maxval=1.0)
    planet_rad = jax.random.uniform(rand_key[3], minval=0.001, maxval=0.25) * ureg.R_sun
    eccentricity = jax.random.uniform(rand_key[4], minval=0.0, maxval=0.9)
    omega = jax.random.uniform(rand_key[5], minval=0.0, maxval=2*jnp.pi)
    Omega = jnp.pi

    # generate jaxoplanet light curve 
    # jaxoplanet works in physical units where the star has mass,
    # can't specify period and semimajor axis
    star = Central(radius=1 * ureg.R_sun, mass=star_mass)
    planet = System(star).add_body(
        time_transit=0.0,
        semimajor=semimajor_axis,
        impact_param=impact_param,
        radius=planet_rad,
        eccentricity=eccentricity,
        omega_peri=omega * ureg.rad,
        asc_node=Omega * ureg.rad,
        mass=0.0
        ).bodies[0]


    jaxoplanet_lc = 1 + limb_dark_light_curve(planet, u, order=100)(t)


    # generate comparison light curve
    Omega = jnp.arctan2(planet.sin_asc_node.to(ureg.radian).magnitude,
        planet.cos_asc_node.to(ureg.radian).magnitude)
    Omega = jnp.where(Omega < 0, Omega + 2*jnp.pi, Omega)

    omega = jnp.arctan2(planet.sin_omega_peri.to(ureg.radian).magnitude,
        planet.cos_omega_peri.to(ureg.radian).magnitude)
    omega = jnp.where(omega < 0, omega + 2*jnp.pi, omega)

    state = {
        "t_peri" : planet.time_peri.to(ureg.day).magnitude,
        "period" : planet.period.to(ureg.day).magnitude,
        "a" :planet.semimajor.to(ureg.R_sun).magnitude,
        "e" : planet.eccentricity.to(ureg.dimensionless).magnitude,
        "i" : planet.inclination.to(ureg.radian).magnitude,
        "Omega" : Omega,
        "omega" : omega,
        "f1" : 0.0, # always circular for testing
        "f2" : 0.0,
        "r" : planet.radius.to(ureg.R_sun).magnitude,
        "obliq" : 0.0,
        "prec" : 0.0,
        "ld_u_coeffs" : jnp.array(u)
    }

    s = OblateSystem(state)
    state = s._state

    test_lc = lightcurve(s._state, t.to(ureg.day).magnitude);

    if not return_lc:
        m = (jaxoplanet_lc != 0) | (test_lc != 0)
        return state, jnp.max(jnp.abs(jaxoplanet_lc - test_lc)), jnp.std(jnp.abs(jaxoplanet_lc[m] - test_lc[m]))
    else:
        return state, jaxoplanet_lc, test_lc

from tqdm import tqdm

states, errs, stds = [], [], []
for i in tqdm(jnp.arange(1000)):
    key = jax.random.key(i)
    s, max_err, std = light_curve_compare(key, 2)
    states.append(s)
    errs.append(e)
    stds.append(std)
errs = jnp.array(errs)