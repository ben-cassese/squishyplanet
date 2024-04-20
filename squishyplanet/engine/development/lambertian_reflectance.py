import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


# for illustration only- not actually doing this 2D integral anywhere
def profile(
    x,
    y,
    p_xx,
    p_xy,
    p_xz,
    p_x0,
    p_yy,
    p_yz,
    p_y0,
    p_zz,
    p_z0,
    p_00,
    x_c,
    y_c,
    z_c,
    **kwargs,
):
    return (
        -(
            x_c
            * (
                p_x0
                + 2 * p_xx * x
                + p_xy * y
                - (
                    p_xz
                    * (
                        p_z0
                        + p_xz * x
                        + p_yz * y
                        - jnp.sqrt(
                            (p_z0 + p_xz * x + p_yz * y) ** 2
                            - 4
                            * p_zz
                            * (
                                -1
                                + p_00
                                + p_x0 * x
                                + p_xx * x**2
                                + p_y0 * y
                                + p_xy * x * y
                                + p_yy * y**2
                            )
                        )
                    )
                )
                / (2.0 * p_zz)
            )
        )
        - (
            p_y0
            + p_xy * x
            + 2 * p_yy * y
            - (
                p_yz
                * (
                    p_z0
                    + p_xz * x
                    + p_yz * y
                    - jnp.sqrt(
                        (p_z0 + p_xz * x + p_yz * y) ** 2
                        - 4
                        * p_zz
                        * (
                            -1
                            + p_00
                            + p_x0 * x
                            + p_xx * x**2
                            + p_y0 * y
                            + p_xy * x * y
                            + p_yy * y**2
                        )
                    )
                )
            )
            / (2.0 * p_zz)
        )
        * y_c
        - jnp.sqrt(
            (p_z0 + p_xz * x + p_yz * y) ** 2
            - 4
            * p_zz
            * (
                -1
                + p_00
                + p_x0 * x
                + p_xx * x**2
                + p_y0 * y
                + p_xy * x * y
                + p_yy * y**2
            )
        )
        * z_c
    ) / (
        jnp.sqrt(
            (p_z0 + p_xz * x + p_yz * y) ** 2
            - 4
            * p_zz
            * (
                -1
                + p_00
                + p_x0 * x
                + p_xx * x**2
                + p_y0 * y
                + p_xy * x * y
                + p_yy * y**2
            )
            + (
                p_x0
                + 2 * p_xx * x
                + p_xy * y
                - (
                    p_xz
                    * (
                        p_z0
                        + p_xz * x
                        + p_yz * y
                        - jnp.sqrt(
                            (p_z0 + p_xz * x + p_yz * y) ** 2
                            - 4
                            * p_zz
                            * (
                                -1
                                + p_00
                                + p_x0 * x
                                + p_xx * x**2
                                + p_y0 * y
                                + p_xy * x * y
                                + p_yy * y**2
                            )
                        )
                    )
                )
                / (2.0 * p_zz)
            )
            ** 2
            + (
                p_y0
                + p_xy * x
                + 2 * p_yy * y
                - (
                    p_yz
                    * (
                        p_z0
                        + p_xz * x
                        + p_yz * y
                        - jnp.sqrt(
                            (p_z0 + p_xz * x + p_yz * y) ** 2
                            - 4
                            * p_zz
                            * (
                                -1
                                + p_00
                                + p_x0 * x
                                + p_xx * x**2
                                + p_y0 * y
                                + p_xy * x * y
                                + p_yy * y**2
                            )
                        )
                    )
                )
                / (2.0 * p_zz)
            )
            ** 2
        )
        * jnp.sqrt(x_c**2 + y_c**2 + z_c**2)
    )
