import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from squishyplanet import OblateSystem

if __name__ == "__main__":
    # start by creating a spherical hot Jupiter
    state = {
        "t_peri": -0.25,
        "times": jnp.linspace(-0.1, 0.1, 500),
        "a": 5.0,
        "period": 1.0,
        "r": 0.1,
        "i": jnp.pi / 2 - 0.1,
        "ld_u_coeffs": jnp.array([0.4, 0.26]),
        "tidally_locked": False,
    }
    spherical_planet = OblateSystem(**state)

    # now we want to create a comparable triaxial planet. let's flatten it along
    # the z and y axes by 10% each
    state["f1"] = 0.1
    state["f2"] = 0.1
    # but, doing that alone would shrink the area of the planet, so we need to scale
    # the radius back up to keep the projected area the same (keep in mind that "r"
    # always refers to the X-axis radius)
    state["r"] = state["r"] / jnp.sqrt(1 - state["f1"])
    triaxial_planet = OblateSystem(**state)
    # note also that after creating an OblateSystem object, you can access the effective
    # radius (radius of a spherical planet with the same instantaneous projected area)
    # through OblateSystem.state["projected_effective_r"]

    # now let's assert that the planet is tidally locked, meaning its projected area
    # will change over the duration of the transit. we again need to scale the radius
    # to keep the projected area comparable
    state["tidally_locked"] = True
    state["r"] = 0.1 / jnp.sqrt(1 - state["f1"]) / jnp.sqrt(1 - state["f2"])
    locked_planet = OblateSystem(**state)

    # now let's create a planet that's not tidally locked, but has some obliquity and
    # precession angle. this will leave us with a fixed ellipse whose projected area
    # will not change over the duration of the transit, but is now "tipped over" a
    # little bit. Since we have a non-zero impact parameter, that tipping will make
    # ingress and egress asymmetrical
    state["tidally_locked"] = False
    state["obliq"] = jnp.pi / 3
    state["prec"] = 0.3
    rolled_planet = OblateSystem(**state)
    eff = rolled_planet.state["projected_effective_r"][0]
    state["r"] *= 0.1 / eff
    rolled_planet = OblateSystem(**state)
    # note that parameterizing a non-tidally locked planet with obliquity and precession
    # is overkill and leads to an over-constrained system, since you can create the same
    # projected ellipse through many combinations of f1, f2, obliq, and prec. However,
    # if the planet is tidally locked, the degeneracy is broken. if dealing with a
    # non-tidally locked planet, it's better to set
    # "parameterize_with_projected_ellipse" to "True", then specify
    # "projected_effective_r", "projected_effective_f", and "projected_effective_theta".

    # let's generate the lightcurves for all of those planets; we don't need any
    # arguments since we aren't changing any parameters
    spherical_lc = spherical_planet.lightcurve()
    triaxial_lc = triaxial_planet.lightcurve()
    locked_lc = locked_planet.lightcurve()
    rolled_lc = rolled_planet.lightcurve()

    # then the rest is plotting

    colors = ["#0082d5", "#b0e384", "#70005e", "#ff7d9c"]

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(6.5, 5))
    axs[0].plot(
        spherical_planet.state["times"] * 24,
        spherical_lc,
        label="spherical",
        c=colors[0],
    )
    axs[0].plot(
        spherical_planet.state["times"] * 24,
        triaxial_lc,
        label="oblate, axes-aligned",
        c=colors[1],
    )
    axs[0].plot(
        spherical_planet.state["times"] * 24,
        locked_lc,
        label="triaxial, tidally locked",
        c=colors[2],
    )
    axs[0].plot(
        spherical_planet.state["times"] * 24,
        rolled_lc,
        label="triaxial, arbitrary orientation",
        c=colors[3],
    )
    axs[0].set(ylim=(0.985, 1.012), ylabel="relative flux\n")
    axs[0].legend(ncols=2, prop={"size": 10}, loc="upper center")

    axs[1].plot(spherical_planet.state["times"] * 24, spherical_lc * 0.0, c=colors[0])
    axs[1].plot(
        spherical_planet.state["times"] * 24,
        (triaxial_lc - spherical_lc) * 1e6,
        c=colors[1],
    )
    axs[1].plot(
        spherical_planet.state["times"] * 24,
        (locked_lc - spherical_lc) * 1e6,
        c=colors[2],
    )
    axs[1].plot(
        spherical_planet.state["times"] * 24,
        (rolled_lc - spherical_lc) * 1e6,
        c=colors[3],
    )

    axs[1].set(
        xlabel="time [hours]",
        ylabel="deviation from\nspherical [ppm]",
        xlim=(-0.07 * 24, 0.07 * 24),
    )
    fig.subplots_adjust(hspace=0)

    plt.savefig("deviations.png", dpi=300, bbox_inches="tight")
