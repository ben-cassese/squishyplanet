import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from squishyplanet import OblateSystem


if __name__ == "__main__":
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

    state["f1"] = 0.1
    state["f2"] = 0.1
    state["r"] = state["r"] / jnp.sqrt(1 - state["f1"])
    triaxial_planet = OblateSystem(**state)

    state["tidally_locked"] = True
    state["r"] = 0.1 / jnp.sqrt(1 - state["f1"]) / jnp.sqrt(1 - state["f2"])
    locked_planet = OblateSystem(**state)

    state["tidally_locked"] = False
    state["obliq"] = jnp.pi / 3
    state["prec"] = 0.3
    rolled_planet = OblateSystem(**state)
    eff = rolled_planet.state["effective_projected_r"][0]
    state["r"] *= 0.1 / eff
    rolled_planet = OblateSystem(**state)

    spherical_lc = spherical_planet.lightcurve()
    triaxial_lc = triaxial_planet.lightcurve()
    locked_lc = locked_planet.lightcurve()
    rolled_lc = rolled_planet.lightcurve()

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
