module intersection_pts
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use model_types, only: rho_coefficients

contains

    function t4_term(rho) result(t4)
        type(rho_coefficients), intent(in) :: rho
        real(dp) :: t4
        t4 = -1._dp + rho%rho_00 - rho%rho_x0 + rho%rho_xx
    end function t4_term

    function t3_term(rho) result(t3)
        type(rho_coefficients), intent(in) :: rho
        real(dp) :: t3
        t3 = -2._dp * rho%rho_xy + 2._dp * rho%rho_y0
    end function t3_term

    function t2_term(rho) result(t2)
        type(rho_coefficients), intent(in) :: rho
        real(dp) :: t2
        t2 = -2._dp + 2._dp * rho%rho_00 - 2._dp * rho%rho_xx + 4._dp * rho%rho_yy
    end function t2_term

    function t1_term(rho) result(t1)
        type(rho_coefficients), intent(in) :: rho
        real(dp) :: t1
        t1 = 2._dp * rho%rho_xy + 2._dp * rho%rho_y0
    end function t1_term

    function t0_term(rho) result(t0)
        type(rho_coefficients), intent(in) :: rho
        real(dp) :: t0
        t0 = -1._dp + rho%rho_00 + rho%rho_x0 + rho%rho_xx
    end function t0_term


    function single_intersection_points(rho) result(intersections)
        type(rho_coefficients), intent(in) :: rho
        real(dp), dimension(4,2) :: intersections
        complex(dp), dimension(4) :: roots_complex
        integer, parameter :: N = 4  ! 4th order polynomial
        real(dp) :: t4, t3, t2, t1, t0
        real(dp), dimension(5) :: coeffs
        complex(dp), dimension(4,4) :: companion
        complex(dp), dimension(4*4) :: work  ! Workspace for ZGEEV
        real(dp), dimension(2*4) :: rwork
        integer :: info
        real(dp), dimension(4) :: t_roots


        t4 = t4_term(rho)
        t3 = t3_term(rho)
        t2 = t2_term(rho)
        t1 = t1_term(rho)
        t0 = t0_term(rho)

        coeffs = [t0, t1, t2, t3, t4]

        ! Normalize coefficients by dividing by leading coefficient
        do i = 1, N
            coeffs(i) = coeffs(i)/coeffs(N+1)
        end do

        ! Construct companion matrix (directly as complex)
        companion = complex(0.0_dp, 0.0_dp)
        do i = 1, N-1
            companion(i+1,i) = complex(1.0_dp, 0.0_dp)
        end do
        do i = 1, N
            companion(i,N) = complex(-coeffs(i), 0.0_dp)
        end do

        ! Find eigenvalues using LAPACK's ZGEEV
        call ZGEEV('N', 'N', N, companion, N, roots_complex, companion, 1, &
                companion, 1, work, 4*N, rwork, info)

        ! now we have the roots of the quartic
        ! figure out which of those are real
        do i=1, 4
            ! ack oof it never turns out to be exactly zero like the jax version,
            ! check if this threshold is good. chance of false positives?
            if (abs(aimag(roots_complex(i))) < 1e-13_dp) then
                t_roots(i) = real(roots_complex(i))
            else
                t_roots(i) = 999._dp
            end if
        end do

        ! convert the t's into xs and ys
        intersections = 999
        do i=1, 4
            if (t_roots(i) == 999._dp) then
                cycle
            end if
            intersections(i,1) = (1 - t_roots(i)*t_roots(i)) / (1 + t_roots(i)*t_roots(i))
            intersections(i,2) = 2 * t_roots(i) / (1 + t_roots(i)*t_roots(i))
        end do

    end function single_intersection_points

end module intersection_pts
