This is a standalone implementation of the "parameterize_with_projected_ellipse" version of the squishyplanet code. The main JAX package does not rely on these routines, and these routines don't rely on the JAX package- this exists only to interface with other Fortran codebases like the original MultiNest and Luna. Though during development we verified it produces outputs for a reasonable set of parameters that match squishyplanet, it is not thoroughly tested or maintained.

It should, fingers crossed, require no external dependencies beyond CMake and a Fortran compiler. It contains local copies of relevant BLAS, LINPACK, and QUADPACK routines.

To run an example version, run:
```bash
mkdir build
cd build
cmake ..
cmake --build .
./squishyplanet
```
