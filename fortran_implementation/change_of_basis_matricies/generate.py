import numpy as np

from squishyplanet.engine.greens_basis_transform import generate_change_of_basis_matrix

for i in range(2, 11):
    m = np.array(generate_change_of_basis_matrix(i))
    # m = np.asfortranarray(m, dtype=np.float64)
    # m.tofile(f"g_matrix_{i}.bin")
    with open(f"g_matrix_{i}.bin", "wb") as f:
        f.write(m.tobytes("F"))  # 'F' ensures Fortran ordering
