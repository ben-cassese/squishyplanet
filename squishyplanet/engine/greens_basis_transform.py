# Taken directly from the repository associated with 
# Agol, Luger, and Foreman-Mackey 2020 (doi:10.3847/1538-3881/ab4fee)
# Many thanks to the authors for making this code available!

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import sympy
from sympy import binomial, symbols, zeros


# Define our symbols
z, n = symbols('z n')

def ptilde(n, z):
    """Return the n^th term in the polynomial basis."""
    return z ** n

def Coefficient(expression, term):
    """Return the coefficient multiplying `term` in `expression`."""
    # Get the coefficient
    coeff = expression.coeff(term)
    # Set any non-constants in this coefficient to zero. If the coefficient
    # is not a constant, this is not the term we are interested in!
    coeff = coeff.subs(z, 0)
    return coeff

def A1(N):
    """Return the change of basis matrix A1."""
    res = zeros(N + 1, N + 1)
    for i in range(N + 1):
        for j in range(N + 1):
            res[i, j] = (-1) ** (i + 1) * binomial(j, i)
    return res

def gtilde(n, z):
    """Return the n^th term in the Green's basis."""
    if n == 0:
        return 1 + 0 * z
    elif n == 1:
        return z
    else:
        return (n + 2) * z ** n - n * z ** (n - 2)

def p_G(n, N):
    """Return the polynomial basis representation of the Green's polynomial `g`."""
    g = gtilde(n, z)
    res = [g.subs(z, 0)]
    for n in range(1, N + 1):
        res.append(Coefficient(g, ptilde(n, z)))
    return res


def A2(N):
    """Return the change of basis matrix A2. The columns of the **inverse** of this matrix are given by `p_G`."""
    res = zeros(N + 1, N + 1)
    for n in range(N + 1):
        res[n] = p_G(n, N)
    return res.inv()

def A(N):
    """Return the full change of basis matrix."""
    return A2(N) * A1(N)

def generate_change_of_basis_matrix(N):
    m = A(N)
    return jnp.array(m, dtype=jnp.float64)


# Forked from jaxoplanet.core.limb_dark.py- this is another way to do it
# that goes around needing to store the change of basis matrix
# @jax.jit
# def greens_basis_transform(u: Array) -> Array:
#     dtype = jnp.dtype(u)
#     u = jnp.concatenate((-jnp.ones(1, dtype=dtype), u))
#     size = len(u)
#     i = np.arange(size)
#     arg = binom(i[None, :], i[:, None]) @ u
#     p = (-1) ** (i + 1) * arg
#     g = [jnp.zeros((), dtype=dtype) for _ in range(size + 2)]
#     for n in range(size - 1, 1, -1):
#         g[n] = p[n] / (n + 2) + g[n + 2]
#     g[1] = p[1] + 3 * g[3]
#     g[0] = p[0] + 2 * g[2]
#     return jnp.stack(g[:-2])