module squishyplanet_2d
   use, intrinsic :: iso_fortran_env, only: dp => real64
   use constants, only: PI, TWO_PI
   use model_types, only: orbit_parameters, planet_parameters_2d, rho_coefficients, skypos_positions, para_coefficients, para_helper_coeffs
   use keplerian, only: kepler, t0_to_t_peri, skypos
   use parametric_ellipse, only: calculate_rho_coefficients, poly_to_parametric, cartesian_intersection_to_parametric_angle
   use intersection_pts, only: intersection_points
   use solution_vecs, only: planet_solution_vec, star_solution_vec
   implicit none

contains

   subroutine squishyplanet_lightcurve_2d( &
      orbit_params, &
      planet_params, &
      ld_u_coeffs, &
      times, &
      change_of_basis_matrix, &
      fluxes &
      )
      implicit none

      ! inputs that change with each sample
      type(orbit_parameters), intent(in) :: orbit_params
      type(planet_parameters_2d), intent(in) :: planet_params
      real(dp), dimension(:), intent(in) :: ld_u_coeffs

      ! inputs that don't change with each sample
      real(dp), dimension(:), intent(in) :: times
      real(dp), dimension(:, :), intent(in) :: change_of_basis_matrix
      real(dp), dimension(:), intent(out) :: fluxes

      ! things that only have to be computed once per lightcurve
      real(dp) :: t_peri
      real(dp) :: area
      real(dp) :: r1
      real(dp) :: r2
      real(dp), dimension(size(ld_u_coeffs) + 1) :: padded_u
      real(dp), allocatable :: g_coeffs(:)
      real(dp) :: normalization_constant

      ! things associated with the loop over times
      integer :: i, q, w, num_intersections
      real(dp) :: tmp1
      real(dp) :: time_delta
      real(dp) :: mean_anomaly
      real(dp) :: true_anomaly
      type(skypos_positions) :: pos
      logical :: possibly_in_transit

      real(dp) :: planet_contribution
      real(dp) :: star_contribution

      ! if it's plausibly transiting
      type(rho_coefficients) :: rho
      type(para_coefficients) :: para
      real(dp), dimension(4, 2) :: pts
      logical :: on_limb

      ! if it's on the limb
      real(dp), dimension(4) :: alphas ! parametric angle on the planet of star intersection

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! compute everything that only has to be done once
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      t_peri = t0_to_t_peri( &
               e=orbit_params%ecc, &
               i=orbit_params%inc, &
               omega=orbit_params%little_omega, &
               period=orbit_params%period, &
               t0=orbit_params%t0)
      area = PI*planet_params%r_eff**2
      r1 = sqrt(area/((1 - planet_params%f_squish_proj)*PI))
      r2 = r1*(1 - planet_params%f_squish_proj)

      padded_u = -1
      padded_u(2:size(ld_u_coeffs) + 1) = ld_u_coeffs
      g_coeffs = matmul(change_of_basis_matrix, padded_u)

      normalization_constant = 1._dp/(PI*(g_coeffs(1) + (2._dp/3._dp)*g_coeffs(2)))

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! the loop over times
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      do i = 1, size(times)

         ! print *, "timestep:", i
         ! print *, "time:", times(i)

         time_delta = times(i) - t_peri
         mean_anomaly = TWO_PI*time_delta/orbit_params%period
         true_anomaly = kepler(M=mean_anomaly, ecc=orbit_params%ecc)

         pos = skypos(orbit_params=orbit_params, f=true_anomaly)

         ! if we're far from the star, don't bother doing anything else
         possibly_in_transit = pos%x**2 + pos%y**2 <= (1.0 + r1*1.1)**2 .and. pos%z > 0
         if (.not. possibly_in_transit) then
            fluxes(i) = 1.0_dp
            ! print *, "not in transit, continuing"
            ! print *, ""
            cycle
         else
            ! print *, "in transit"
         end if

         ! this one introduces a numerical difference w/ the jax version
         rho = calculate_rho_coefficients( &
               projected_r=r1, &
               projected_f=planet_params%f_squish_proj, &
               projected_theta=planet_params%theta_proj, &
               xc=pos%x, &
               yc=pos%y)

         para = poly_to_parametric(rho=rho)

         pts = intersection_points(rho=rho)

         ! check if we're on the limb
         on_limb = .false.
         num_intersections = 0
         do q = 1, 4
            if (pts(q, 1) .ne. 999._dp) then
               num_intersections = num_intersections + 1
            end if
         end do
         on_limb = num_intersections > 0
         ! print *, "num_intersections:", num_intersections

         ! if we're not on the limb, we're either fully inside the star
         if (on_limb .eqv. .false.) then
            ! print *, "no intersections, so fully out or in transit"

            call not_on_limb( &
               para=para, &
               g_coeffs=g_coeffs, &
               normalization_constant=normalization_constant, &
               planet_contribution=planet_contribution, &
               star_contribution=star_contribution &
               )

            ! if we're on the limb
         else
            ! print *, "on the limb"

            ! convert the x,y positions of the intersection to
            ! parametric angles wrt the planet
            alphas = cartesian_intersection_to_parametric_angle( &
                     xs=pts(:, 1), &
                     ys=pts(:, 2), &
                     para=para &
                     )
            ! filter them to only look at the ones corresponding to real intersections
            do q = 1, 4
               if (pts(q, 1) .eq. 999._dp) then
                  alphas(q) = TWO_PI
               end if
            end do

            ! housekeeping to wrap them to [0, 2*pi)
            do q = 1, 4
               if (alphas(q) .eq. 999._dp) then
                  alphas(q) = TWO_PI
               end if
               if (alphas(q) .lt. 0.0_dp) then
                  alphas(q) = alphas(q) + TWO_PI
               end if
               if (alphas(q) .gt. TWO_PI) then
                  alphas(q) = alphas(q) - TWO_PI
               end if
            end do

            ! sort them
            do q = 1, 3
               do w = 1, 4 - q
                  if (alphas(w) > alphas(w + 1)) then
                     ! Swap elements
                     tmp1 = alphas(w)
                     alphas(w) = alphas(w + 1)
                     alphas(w + 1) = tmp1
                  end if
               end do
            end do

            if (num_intersections == 2) then
               ! print *, "two intersections, calling subroutine"
               call two_intersections( &
                  alpha_1=alphas(1), &
                  alpha_2=alphas(2), &
                  para=para, &
                  g_coeffs=g_coeffs, &
                  normalization_constant=normalization_constant, &
                  planet_contribution=planet_contribution, &
                  star_contribution=star_contribution &
                  )
            else
               ! print *, "four intersections"
               call four_intersections( &
                  alphas=alphas, &
                  para=para, &
                  g_coeffs=g_coeffs, &
                  normalization_constant=normalization_constant, &
                  planet_contribution=planet_contribution, &
                  star_contribution=star_contribution &
                  )
            end if
         end if ! on_limb

         fluxes(i) = 1.0_dp - (planet_contribution + star_contribution)
         ! print *, "fluxes(i):", fluxes(i)
      end do

   end subroutine squishyplanet_lightcurve_2d

   subroutine not_on_limb( &
      para, &
      g_coeffs, &
      normalization_constant, &
      planet_contribution, &
      star_contribution &
      )
      implicit none
      type(para_coefficients), intent(in) :: para
      real(dp), allocatable, intent(in) :: g_coeffs(:)
      real(dp), intent(in) :: normalization_constant
      real(dp), intent(out) :: planet_contribution
      real(dp), intent(out) :: star_contribution

      real(dp), allocatable :: planet_solution_vector(:)
      logical :: fully_inside_star

      ! first, we're not actually sure if we're inside the star or not:
      ! we only know we're not on the limb
      fully_inside_star = (para%c_x3*para%c_x3 + para%c_y3*para%c_y3) <= 1

      if (fully_inside_star .neqv. .true.) then
         ! print *, "actually, not in transit, fooled by the buffer"
         planet_contribution = 0.0_dp
         star_contribution = 0.0_dp
         return
      end if

      ! print *, "ok, we're fully inside the star, integrating just the planet"

      ! if we're fully inside the star, we just need to integrate the planet
      ! from 0 to 2*pi
      call planet_solution_vec( &
         a=0.0_dp, &
         b=TWO_PI, &
         g_coeffs=g_coeffs, &
         para=para, &
         solution_vector=planet_solution_vector &
         )

      planet_contribution = dot_product(g_coeffs, planet_solution_vector)*normalization_constant
      star_contribution = 0.0_dp

   end subroutine not_on_limb

   subroutine two_intersections( &
      alpha_1, &
      alpha_2, &
      para, &
      g_coeffs, &
      normalization_constant, &
      planet_contribution, &
      star_contribution &
      )
      implicit none
      real(dp), intent(in) :: alpha_1
      real(dp), intent(in) :: alpha_2
      type(para_coefficients), intent(in) :: para
      real(dp), allocatable, intent(in) :: g_coeffs(:)
      real(dp), intent(in) :: normalization_constant
      real(dp), intent(out) :: planet_contribution
      real(dp), intent(out) :: star_contribution

      real(dp), allocatable :: planet_solution_vector(:), star_solution_vector(:)
      real(dp) :: test_ang, test_val, tmp1, tmp2

      planet_contribution = 0.0_dp
      star_contribution = 0.0_dp
      ! print *, "alpha_1:", alpha_1
      ! print *, "alpha_2:", alpha_2

      ! ! check the orientation of the planet
      test_ang = alpha_1 + (alpha_2 - alpha_1)/2.0_dp
      if (test_ang > TWO_PI) then
         test_ang = test_ang - TWO_PI
      end if
      tmp1 = dcos(test_ang)
      tmp2 = dsin(test_ang)

      test_val = sqrt( &
                 (para%c_x1*tmp1 + para%c_x2*tmp2 + para%c_x3)**2 + &
                 (para%c_y1*tmp1 + para%c_y2*tmp2 + para%c_y3)**2 &
                 )

      if (test_val > 1.0_dp) then
         ! print *, "test_val is outside the star"
         ! if you're outside the star, instead of integrating two legs separately,
         ! you can just wrap passed 2pi
         call planet_solution_vec( &
            a=alpha_2, &
            b=alpha_1 + TWO_PI, &
            g_coeffs=g_coeffs, &
            para=para, &
            solution_vector=planet_solution_vector &
            )
         planet_contribution = dot_product(g_coeffs, planet_solution_vector)*normalization_constant
      else
         ! print *, "test_val is inside the star"
         call planet_solution_vec( &
            a=alpha_1, &
            b=alpha_2, &
            g_coeffs=g_coeffs, &
            para=para, &
            solution_vector=planet_solution_vector &
            )
         planet_contribution = dot_product(g_coeffs, planet_solution_vector)*normalization_constant
      end if

      ! regardless, always integrate the star from alpha1 to alpha2
      ! (alpha is defined wrt the planet, but star_solution_vec converts it to the star's frame)
      call star_solution_vec( &
         a=alpha_1, &
         b=alpha_2, &
         g_coeffs=g_coeffs, &
         para=para, &
         solution_vector=star_solution_vector &
         )
      star_contribution = dot_product(g_coeffs, star_solution_vector)*normalization_constant

      ! print *, "planet_contribution:", planet_contribution
      ! print *, "star_contribution:", star_contribution

   end subroutine two_intersections

   subroutine four_intersections( &
      alphas, &
      para, &
      g_coeffs, &
      normalization_constant, &
      planet_contribution, &
      star_contribution &
      )
      implicit none
      real(dp), intent(in) :: alphas(4)
      type(para_coefficients), intent(in) :: para
      real(dp), allocatable, intent(in) :: g_coeffs(:)
      real(dp), intent(in) :: normalization_constant
      real(dp), intent(out) :: planet_contribution
      real(dp), intent(out) :: star_contribution

      integer :: i
      real(dp), dimension(4, 2) :: alpha_pairs
      real(dp) :: a1, a2, test_ang, test_val, tmp1, tmp2
      logical :: is_planet_chunk
      real(dp), allocatable :: planet_solution_vector(:), star_solution_vector(:)

      star_contribution = 0.0_dp
      planet_contribution = 0.0_dp

      do i = 1, 4
         alpha_pairs(i, 1) = alphas(i)
         alpha_pairs(i, 2) = alphas(mod(i, 4) + 1)
      end do

      do i = 1, 4
         a1 = alpha_pairs(i, 1)
         a2 = alpha_pairs(i, 2)

         ! figure out if we're looking at a chunk that's along the edge
         ! of the planet or the star
         test_ang = a1 + (a2 - a1)/2.0_dp
         if (test_ang > TWO_PI) then
            test_ang = test_ang - TWO_PI
         end if
         tmp1 = dcos(test_ang)
         tmp2 = dsin(test_ang)

         test_val = sqrt( &
                    (para%c_x1*tmp1 + para%c_x2*tmp2 + para%c_x3)**2 + &
                    (para%c_y1*tmp1 + para%c_y2*tmp2 + para%c_y3)**2 &
                    )
         is_planet_chunk = test_val < 1.0_dp

         if (is_planet_chunk) then
            call planet_solution_vec( &
               a=a1, &
               b=a2, &
               g_coeffs=g_coeffs, &
               para=para, &
               solution_vector=planet_solution_vector &
               )
            planet_contribution = planet_contribution + dot_product(g_coeffs, planet_solution_vector)*normalization_constant
            star_contribution = star_contribution + 0.0_dp
         else
            call star_solution_vec( &
               a=a1, &
               b=a2, &
               g_coeffs=g_coeffs, &
               para=para, &
               solution_vector=star_solution_vector &
               )
            planet_contribution = planet_contribution + 0.0_dp
            star_contribution = star_contribution + dot_product(g_coeffs, star_solution_vector)*normalization_constant
         end if
      end do

   end subroutine four_intersections

end module squishyplanet_2d
