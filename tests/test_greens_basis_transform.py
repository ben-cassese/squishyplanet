import numpy as np
import pytest
from sympy import Matrix, binomial, symbols, zeros

from squishyplanet.engine.greens_basis_transform import generate_change_of_basis_matrix


def _sympy_change_of_basis_matrix(order: int) -> Matrix:
    """Reference from the original symbolic implementation, before the NumPy rewrite."""
    z = symbols("z")
    a1 = zeros(order + 1, order + 1)
    inverse_a2 = zeros(order + 1, order + 1)
    for i in range(order + 1):
        for j in range(order + 1):
            a1[i, j] = (-1) ** (i + 1) * binomial(j, i)

    for n in range(order + 1):
        if n == 0:
            g = 1 + 0 * z
        elif n == 1:
            g = z
        else:
            g = (n + 2) * z**n - n * z ** (n - 2)
        coefficients = [g.subs(z, 0)]
        for k in range(1, order + 1):
            coefficients.append(g.coeff(z**k).subs(z, 0))
        inverse_a2[n] = coefficients
    return inverse_a2.inv() * a1


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 8, 16, 32, 50])
def test_numpy_basis_matches_sympy(order: int) -> None:
    """Compare every matrix entry with an exact symbolic reference, rounded to float64."""
    expected = np.array(_sympy_change_of_basis_matrix(order), dtype=np.float64)
    actual = np.asarray(generate_change_of_basis_matrix(order))
    assert actual.dtype == np.float64
    # High-order entries grow rapidly: use entrywise relative error, rather than
    # scaling the tolerance by the largest entry in the entire matrix.
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
