module keplerian
   use iso_fortran_env, only: dp => real64
   use constants, only: PI, TWO_PI
   use model_types
   implicit none

contains
   function true_anomaly_at_transit_center(e, i, omega) result(true_anomaly)
      implicit none

      ! Input variables (all scalar)
      real(dp), intent(in) :: e, i, omega
      ! Output variable
      real(dp) :: true_anomaly

      ! Local variables
      real(dp) :: hp, kp, dcos_i_squared
      real(dp) :: eta_1, eta_2, eta_3, eta_4, eta_5, eta_6

      ! Calculate intermediate values
      hp = e*dsin(omega)
      kp = e*dcos(omega)
      dcos_i_squared = dcos(i)**2

      ! Calculate η₁ term
      eta_1 = (kp/(1.0_dp + hp))*dcos_i_squared

      ! Calculate η₂ term
      eta_2 = (kp/(1.0_dp + hp))* &
              (1.0_dp/(1.0_dp + hp))* &
              dcos_i_squared**2

      ! Calculate η₃ term
      eta_3 = -(kp/(1.0_dp + hp))* &
              ((-6.0_dp*(1.0_dp + hp) + &
                kp**2*(-1.0_dp + 2.0_dp*hp))/ &
               (6.0_dp*(1.0_dp + hp)**3))* &
              dcos_i_squared**3

      ! Calculate η₄ term
      eta_4 = -(kp/(1.0_dp + hp))* &
              ((-2.0_dp*(1.0_dp + hp) + &
                kp**2*(-1.0_dp + 3.0_dp*hp))/ &
               (2.0_dp*(1.0_dp + hp)**4))* &
              dcos_i_squared**4

      ! Calculate η₅ term
      eta_5 = (kp/(1.0_dp + hp))* &
              ((40.0_dp*(1.0_dp + hp)**2 - &
                40.0_dp*kp**2*(-1.0_dp + 3.0_dp*hp + 4.0_dp*hp**2) + &
                kp**4*(3.0_dp - 19.0_dp*hp + 8.0_dp*hp**2))/ &
               (40.0_dp*(1.0_dp + hp)**6))* &
              dcos_i_squared**5

      ! Calculate η₆ term
      eta_6 = (kp/(1.0_dp + hp))* &
              ((24.0_dp*(1.0_dp + hp)**2 - &
                40.0_dp*kp**2*(-1.0_dp + 4.0_dp*hp + 5.0_dp*hp**2) + &
                9.0_dp*kp**4*(1.0_dp - 8.0_dp*hp + 5.0_dp*hp**2))/ &
               (24.0_dp*(1.0_dp + hp)**7))* &
              dcos_i_squared**6

      ! Calculate final true anomaly
      true_anomaly = PI/2.0_dp - omega - eta_1 - eta_2 - eta_3 - eta_4 - eta_5 - eta_6

   end function true_anomaly_at_transit_center

   function t0_to_t_peri(e, i, omega, period, t0) result(t_peri)
      implicit none
      real(dp), intent(in) :: e, i, omega, period, t0
      real(dp) :: t_peri

      real(dp) :: f, eccentric_anomaly, mean_anomaly

      f = true_anomaly_at_transit_center(e, i, omega)

      eccentric_anomaly = atan2(sqrt(1 - e**2)*dsin(f), e + dcos(f))
      mean_anomaly = eccentric_anomaly - e*dsin(eccentric_anomaly)

      t_peri = t0 - period/(2*PI)*mean_anomaly

   end function t0_to_t_peri

   function kepler(M, ecc) result(f)
      ! Solve Kepler's equation to compute the true anomaly
      ! Args:
      !   M (dp): Mean anomaly in radians
      !   ecc (dp): Eccentricity (dimensionless)
      ! Returns:
      !   f (dp): True anomaly in radians [0, 2π)
      implicit none

      real(dp), intent(in) :: M, ecc
      real(dp) :: f, dsinf, dcosf

      call kepler_internal(M, ecc, dsinf, dcosf)

      ! Calculate arctangent and ensure result is in [0, 2π)
      f = atan2(dsinf, dcosf)
      if (f < 0.0_dp) then
         f = f + TWO_PI
      end if
   end function kepler

   subroutine kepler_internal(M, ecc, dsinf, dcosf)
      implicit none
      real(dp), intent(in) :: M, ecc
      real(dp), intent(out) :: dsinf, dcosf
      real(dp) :: M_wrapped, E, ome
      logical :: high
      real(dp) :: tan_half_f, tan2_half_f, denom

      ! Wrap into the right range
      M_wrapped = modulo(M, TWO_PI)

      ! We can restrict to the range [0, PI)
      high = M_wrapped > PI
      if (high) then
         M_wrapped = TWO_PI - M_wrapped
      end if

      ! Solve
      ome = 1.0_dp - ecc
      E = starter(M_wrapped, ecc, ome)
      E = refine(M_wrapped, ecc, ome, E)

      ! Re-wrap back into the full range
      if (high) then
         E = TWO_PI - E
      end if

      ! Convert to true anomaly
      tan_half_f = sqrt((1.0_dp + ecc)/(1.0_dp - ecc))*tan(0.5_dp*E)
      tan2_half_f = tan_half_f*tan_half_f

      ! Compute dsin(f) and dcos(f)
      denom = 1.0_dp/(1.0_dp + tan2_half_f)
      dsinf = 2.0_dp*tan_half_f*denom
      dcosf = (1.0_dp - tan2_half_f)*denom
   end subroutine kepler_internal

   function starter(M, ecc, ome) result(E)
      implicit none
      real(dp), intent(in) :: M, ecc, ome
      real(dp) :: E
      real(dp) :: M2, alpha, d, alphad, r, q, q2, w

      M2 = M*M
      alpha = 3.0_dp*PI/(PI - 6.0_dp/PI)
      alpha = alpha + 1.6_dp/(PI - 6.0_dp/PI)*(PI - M)/(1.0_dp + ecc)

      d = 3.0_dp*ome + alpha*ecc
      alphad = alpha*d
      r = (3.0_dp*alphad*(d - ome) + M2)*M
      q = 2.0_dp*alphad*ome - M2
      q2 = q*q

      w = (abs(r) + sqrt(q2*q + r*r))**(1.0_dp/3.0_dp)
      w = w*w

      E = (2.0_dp*r*w/(w*w + w*q + q2) + M)/d
   end function starter

   function refine(M, ecc, ome, E_in) result(E_out)
      implicit none
      real(dp), intent(in) :: M, ecc, ome, E_in
      real(dp) :: E_out
      real(dp) :: sE, cE, f_0, f_1, f_2, f_3
      real(dp) :: d_3, d_4, d_42, dE

      sE = E_in - dsin(E_in)
      cE = 1.0_dp - dcos(E_in)

      f_0 = ecc*sE + E_in*ome - M
      f_1 = ecc*cE + ome
      f_2 = ecc*(E_in - sE)
      f_3 = 1.0_dp - f_1

      d_3 = -f_0/(f_1 - 0.5_dp*f_0*f_2/f_1)
      d_4 = -f_0/(f_1 + 0.5_dp*d_3*f_2 + (d_3*d_3)*f_3/6.0_dp)
      d_42 = d_4*d_4

      dE = -f_0/(f_1 + 0.5_dp*d_4*f_2 + d_4*d_4*f_3/6.0_dp - &
                 d_42*d_4*f_2/24.0_dp)

      E_out = E_in + dE

   end function refine

   function x_position(a, e, f, big_Omega, i, little_omega) result(x)
      ! Compute x coordinate in the sky frame
      ! Args:
      !   a (dp): Semi-major axis in stellar radii
      !   e (dp): Eccentricity
      !   f (dp): True anomaly in radians
      !   big_Omega (dp): Longitude of ascending node (Ω) in radians
      !   i (dp): Inclination in radians
      !   little_omega (dp): Argument of periapsis (ω) in radians
      implicit none
      real(dp), intent(in) :: a, e, f, big_Omega, i, little_omega
      real(dp) :: x
      real(dp) :: factor1, factor2, term1, term2

      ! Pre-compute common factors to improve readability and efficiency
      factor1 = a*(1.0_dp - e*e)/(1.0_dp + e*dcos(f))
      factor2 = dsin(f)

      ! First parenthesized term
      term1 = factor2*(dcos(big_Omega)*dsin(little_omega) + &
                       dcos(i)*dcos(little_omega)*dsin(big_Omega))

      ! Second parenthesized term
      term2 = dcos(f)*(-dcos(little_omega)*dcos(big_Omega) + &
                       dcos(i)*dsin(little_omega)*dsin(big_Omega))

      x = -factor1*(term1 + term2)
   end function x_position

   function y_position(a, e, f, big_Omega, i, little_omega) result(y)
      ! Compute y coordinate in the sky frame
      ! Args similar to x_position
      implicit none
      real(dp), intent(in) :: a, e, f, big_Omega, i, little_omega
      real(dp) :: y
      real(dp) :: factor, combined_angle

      factor = a*(1.0_dp - e*e)/(1.0_dp + e*dcos(f))
      combined_angle = f + little_omega

      y = factor*(dcos(i)*dcos(big_Omega)*dsin(combined_angle) + &
                  dcos(combined_angle)*dsin(big_Omega))
   end function y_position

   function z_position(a, e, f, big_Omega, i, little_omega) result(z)
      ! Compute z coordinate in the sky frame
      ! Args similar to x_position
      implicit none
      real(dp), intent(in) :: a, e, f, big_Omega, i, little_omega
      real(dp) :: z
      real(dp) :: factor

      factor = a*(1.0_dp - e*e)/(1.0_dp + e*dcos(f))
      z = factor*dsin(i)*dsin(f + little_omega)
   end function z_position

   function skypos(orbit_params, f) result(pos)
      ! Compute the sky position of the planet
      ! Args:
      !   orbit_params (model_parameters): Model parameters
      !   f (dp): True anomaly in radians
      ! Returns:
      !   pos (skypos_positions): Sky position of the planet
      implicit none
      type(orbit_parameters) :: orbit_params
      real(dp):: f
      type(skypos_positions) :: pos

      pos%x = x_position(orbit_params%semi, orbit_params%ecc, f, &
                         orbit_params%big_Omega, orbit_params%inc, orbit_params%little_omega)
      pos%y = y_position(orbit_params%semi, orbit_params%ecc, f, &
                         orbit_params%big_Omega, orbit_params%inc, orbit_params%little_omega)
      pos%z = z_position(orbit_params%semi, orbit_params%ecc, f, &
                         orbit_params%big_Omega, orbit_params%inc, orbit_params%little_omega)
   end function skypos

end module keplerian
