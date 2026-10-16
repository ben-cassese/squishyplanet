# Taken directly from the repository associated with
# Agol, Luger, and Foreman-Mackey 2020 (doi:10.3847/1538-3881/ab4fee)
# Many thanks to the authors for making this code available!

import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def generate_change_of_basis_matrix(N: int) -> jax.Array:
    """Generate the change of basis matrix to convert limb darking u coefficients to
    Green's basis coefficients.

    This function is only run once per system, though the resulting matrix is used
    repeatedly in the light curve calculation. It implements Eq. 17 of `Agol, Luger, and
    Foreman-Mackey 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A/abstract>`_.

    Args:
        N (int): The order of the polynomial limb darkening law.

    Returns:
        Array:
            The change of basis matrix. When solving for the blocked flux, this will
            be multiplied by the u limb darkening coefficients to convert them to the
            Green's basis.

    """
    # A1: the u -> polynomial basis transform, entry (i, j) = (-1)^(i+1) * binom(j, i)
    i = np.arange(N + 1)
    A1 = (-1.0) ** (i[:, None] + 1) * np.array(
        [[math.comb(j, ii) for j in range(N + 1)] for ii in i], dtype=float
    )
    # G: the Green's basis vectors gtilde_n written in the polynomial basis, where
    # gtilde_0 = 1, gtilde_1 = z, gtilde_n = (n + 2) z^n - n z^(n - 2)
    G = np.eye(N + 1)
    for n in range(2, N + 1):
        G[n, n], G[n, n - 2] = n + 2, -n
    # A2 = inv(G.T), so the full transform A2 @ A1 is a single solve
    return jnp.array(np.linalg.solve(G.T, A1), dtype=jnp.float64)
