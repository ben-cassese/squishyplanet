module three_d_coefficients
   use iso_fortran_env, only: dp => real64
   use constants, only: PI
   use model_types, only: orbit_parameters, planet_parameters_3d, p_coefficients, rho_coefficients
   implicit none

contains
   subroutine compute_3d_coeffs(orbit_params, planet_params, true_anomaly, p_coeffs)
      implicit none
      type(orbit_parameters), intent(in) :: orbit_params
      type(planet_parameters_3d), intent(in) :: planet_params
      real(dp), intent(in) :: true_anomaly
      type(p_coefficients), intent(out) :: p_coeffs

      real(dp) :: sin_little_omega, cos_little_omega
      real(dp) :: sin_big_Omega, cos_big_Omega
      real(dp) :: sin_inc, cos_inc
      real(dp) :: cos_ta

      real(dp) :: sin_obliquity, cos_obliquity
      real(dp) :: sin_precession, cos_precession

      real(dp) :: sin_2_obliquity, cos_2_obliquity
      real(dp) :: sin_2_big_Omega, cos_2_big_Omega
      real(dp) :: sin_precession_plus_little_omega, cos_precession_plus_little_omega
      real(dp) :: sin_2_little_omega, cos_2_little_omega
      real(dp) :: sin_2_precession
      real(dp) :: sin_ta_minus_precession, cos_ta_minus_precession
      real(dp) :: sin_omega_plus_Omega, cos_omega_plus_Omega

      ! this is lazy but... it's a lot of variables
      real(dp) :: a, e, i, little_omega, big_Omega, f1, f2, r, f, precession, obliquity
      a = orbit_params%semi
      e = orbit_params%ecc
      i = orbit_params%inc
      little_omega = orbit_params%little_omega
      big_Omega = orbit_params%big_Omega
      f1 = planet_params%f_squish_1
      f2 = planet_params%f_squish_2
      r = planet_params%r
      precession = planet_params%precession
      obliquity = planet_params%obliquity
      f = true_anomaly

      sin_little_omega = dsin(orbit_params%little_omega)
      cos_little_omega = dcos(orbit_params%little_omega)
      sin_big_Omega = dsin(orbit_params%big_Omega)
      cos_big_Omega = dcos(orbit_params%big_Omega)
      sin_inc = dsin(orbit_params%inc)
      cos_inc = dcos(orbit_params%inc)
      cos_ta = dcos(true_anomaly)

      sin_obliquity = dsin(planet_params%obliquity)
      cos_obliquity = dcos(planet_params%obliquity)
      sin_precession = dsin(planet_params%precession)
      cos_precession = dcos(planet_params%precession)

      sin_2_big_Omega = dsin(2.0_dp*orbit_params%big_Omega)
      cos_2_big_Omega = dcos(2.0_dp*orbit_params%big_Omega)
      sin_2_little_omega = dsin(2.0_dp*orbit_params%little_omega)
      cos_2_little_omega = dcos(2.0_dp*orbit_params%little_omega)
      sin_omega_plus_Omega = dsin(orbit_params%little_omega + orbit_params%big_Omega)
      cos_omega_plus_Omega = dcos(orbit_params%little_omega + orbit_params%big_Omega)
      sin_precession_plus_little_omega = dsin(planet_params%precession + orbit_params%little_omega)
      cos_precession_plus_little_omega = dcos(planet_params%precession + orbit_params%little_omega)

      sin_2_obliquity = dsin(2.0_dp*planet_params%obliquity)
      cos_2_obliquity = dcos(2.0_dp*planet_params%obliquity)
      sin_2_precession = dsin(2.0_dp*planet_params%precession)
      sin_ta_minus_precession = dsin(true_anomaly - planet_params%precession)
      cos_ta_minus_precession = dcos(true_anomaly - planet_params%precession)

      p_coeffs%p_xx = ((cos_little_omega*(cos_big_Omega*sin_precession + cos_inc*cos_precession*sin_big_Omega) + sin_little_omega*(cos_precession*cos_big_Omega - cos_inc*sin_precession*sin_big_Omega))**2/(-1 + f2)**2 + (sin_inc*sin_obliquity*sin_big_Omega + cos_obliquity*sin_precession*(cos_big_Omega*sin_little_omega + cos_inc*cos_little_omega*sin_big_Omega) + cos_precession*cos_obliquity*(-(cos_little_omega*cos_big_Omega) + cos_inc*sin_little_omega*sin_big_Omega))**2 + (cos_big_Omega*sin_precession*sin_obliquity*sin_little_omega + (-(cos_obliquity*sin_inc) + cos_inc*cos_little_omega*sin_precession*sin_obliquity)*sin_big_Omega + cos_precession*sin_obliquity*(-(cos_little_omega*cos_big_Omega) + cos_inc*sin_little_omega*sin_big_Omega))**2/(-1 + f1)**2)/r**2
      p_coeffs%p_xy = ((16*(cos_big_Omega**2*sin_inc*sin_precession*sin_2_obliquity*sin_little_omega - 2*cos_obliquity**2*cos_big_Omega*sin_inc**2*sin_big_Omega - 2*cos_obliquity*sin_inc*sin_precession*sin_obliquity*sin_little_omega*sin_big_Omega**2 + cos_inc*sin_obliquity*(cos_2_little_omega*cos_2_big_Omega*sin_2_precession*sin_obliquity + cos_precession**2*cos_2_big_Omega*sin_obliquity*sin_2_little_omega - cos_2_big_Omega*sin_precession**2*sin_obliquity*sin_2_little_omega + 4*cos_obliquity*cos_little_omega*cos_big_Omega*sin_inc*sin_precession*sin_big_Omega + 4*cos_precession*cos_obliquity*cos_big_Omega*sin_inc*sin_little_omega*sin_big_Omega) - cos_precession*cos_little_omega*(cos_2_big_Omega*sin_inc*sin_2_obliquity + 4*cos_big_Omega*sin_precession*sin_obliquity**2*sin_little_omega*sin_big_Omega) + cos_precession**2*cos_little_omega**2*sin_obliquity**2*sin_2_big_Omega + sin_precession**2*sin_obliquity**2*sin_little_omega**2*sin_2_big_Omega - cos_inc**2*sin_obliquity**2*sin_precession_plus_little_omega**2*sin_2_big_Omega))/(-1 + f1)**2 - 32*(-(cos_precession**2*cos_obliquity**2*cos_little_omega**2*cos_big_Omega*sin_big_Omega) + cos_big_Omega*sin_inc**2*sin_obliquity**2*sin_big_Omega + cos_precession*cos_obliquity*cos_little_omega*(-(cos_big_Omega**2*sin_inc*sin_obliquity) - cos_inc*cos_obliquity*cos_2_big_Omega*sin_precession_plus_little_omega + sin_inc*sin_obliquity*sin_big_Omega**2 + cos_obliquity*sin_precession*sin_little_omega*sin_2_big_Omega) + cos_obliquity*sin_inc*sin_obliquity*(cos_big_Omega**2*sin_precession*sin_little_omega - sin_precession*sin_little_omega*sin_big_Omega**2 + cos_inc*sin_precession_plus_little_omega*sin_2_big_Omega) + cos_obliquity**2*(cos_inc*cos_2_big_Omega*sin_precession*sin_little_omega*sin_precession_plus_little_omega - cos_big_Omega*sin_precession**2*sin_little_omega**2*sin_big_Omega + (cos_inc**2*sin_precession_plus_little_omega**2*sin_2_big_Omega)/2.)) + (4*Sin(i - 2*(precession + little_omega - big_Omega)) - 4*Sin(i + 2*(precession + little_omega - big_Omega)) + 2*Sin(2*(i - big_Omega)) + Sin(2*(i - precession - little_omega - big_Omega)) + 6*Sin(2*(precession + little_omega - big_Omega)) + Sin(2*(i + precession + little_omega - big_Omega)) + 4*sin_2_big_Omega - 2*Sin(2*(i + big_Omega)) - Sin(2*(i - precession - little_omega + big_Omega)) - 6*Sin(2*(precession + little_omega + big_Omega)) - Sin(2*(i + precession + little_omega + big_Omega)) + 4*Sin(i - 2*(precession + little_omega + big_Omega)) - 4*Sin(i + 2*(precession + little_omega + big_Omega)))/(-1 + f2)**2)/(16.*r**2)
      p_coeffs%p_xz = (2*(-((cos_precession_plus_little_omega*sin_inc*(cos_little_omega*(cos_big_Omega*sin_precession + cos_inc*cos_precession*sin_big_Omega) + sin_little_omega*(cos_precession*cos_big_Omega - cos_inc*sin_precession*sin_big_Omega)))/(-1 + f2)**2) + (cos_inc*sin_obliquity - cos_obliquity*sin_inc*sin_precession_plus_little_omega)*(sin_inc*sin_obliquity*sin_big_Omega + cos_obliquity*sin_precession*(cos_big_Omega*sin_little_omega + cos_inc*cos_little_omega*sin_big_Omega) + cos_precession*cos_obliquity*(-(cos_little_omega*cos_big_Omega) + cos_inc*sin_little_omega*sin_big_Omega)) - ((cos_inc*cos_obliquity + sin_inc*sin_obliquity*sin_precession_plus_little_omega)*(cos_big_Omega*sin_precession*sin_obliquity*sin_little_omega + (-(cos_obliquity*sin_inc) + cos_inc*cos_little_omega*sin_precession*sin_obliquity)*sin_big_Omega + cos_precession*sin_obliquity*(-(cos_little_omega*cos_big_Omega) + cos_inc*sin_little_omega*sin_big_Omega)))/(-1 + f1)**2))/r**2
      p_coeffs%p_x0 = (2*a*(-1 + e**2)*(-((sin_ta_minus_precession*(cos_little_omega*(cos_big_Omega*sin_precession + cos_inc*cos_precession*sin_big_Omega) + sin_little_omega*(cos_precession*cos_big_Omega - cos_inc*sin_precession*sin_big_Omega)))/(-1 + f2)**2) + (cos_ta_minus_precession*(cos_precession*(2 - 2*f1 + f1**2 + (-2 + f1)*f1*cos_2_obliquity)*(cos_little_omega*cos_big_Omega - cos_inc*sin_little_omega*sin_big_Omega) - 2*((-2 + f1)*f1*cos_obliquity*sin_inc*sin_obliquity*sin_big_Omega + (-1 + f1)**2*cos_obliquity**2*sin_precession*(cos_big_Omega*sin_little_omega + cos_inc*cos_little_omega*sin_big_Omega) + sin_precession*sin_obliquity**2*(cos_big_Omega*sin_little_omega + cos_inc*cos_little_omega*sin_big_Omega))))/(2.*(-1 + f1)**2)))/(r**2*(1 + e*cos_ta))
      p_coeffs%p_yy = ((cos_big_Omega*(sin_inc*sin_obliquity + cos_inc*cos_obliquity*sin_precession_plus_little_omega) + cos_obliquity*cos_precession_plus_little_omega*sin_big_Omega)**2 + (cos_inc*cos_precession_plus_little_omega*cos_big_Omega - sin_precession_plus_little_omega*sin_big_Omega)**2/(-1 + f2)**2 + (cos_obliquity*cos_big_Omega*sin_inc - sin_obliquity*(cos_inc*cos_big_Omega*sin_precession_plus_little_omega + cos_precession_plus_little_omega*sin_big_Omega))**2/(-1 + f1)**2)/r**2
      p_coeffs%p_yz = (2*(-((cos_inc*sin_obliquity - cos_obliquity*sin_inc*sin_precession_plus_little_omega)*(cos_big_Omega*(sin_inc*sin_obliquity + cos_inc*cos_obliquity*sin_precession_plus_little_omega) + cos_obliquity*cos_precession_plus_little_omega*sin_big_Omega)) + (cos_precession_plus_little_omega*sin_inc*(cos_inc*cos_precession_plus_little_omega*cos_big_Omega - sin_precession_plus_little_omega*sin_big_Omega))/(-1 + f2)**2 + ((cos_inc*cos_obliquity + sin_inc*sin_obliquity*sin_precession_plus_little_omega)*(-(cos_obliquity*cos_big_Omega*sin_inc) + sin_obliquity*(cos_inc*cos_big_Omega*sin_precession_plus_little_omega + cos_precession_plus_little_omega*sin_big_Omega)))/(-1 + f1)**2))/r**2
      p_coeffs%p_y0 = (2*a*(-1 + e**2)*(cos_ta_minus_precession*cos_obliquity*(cos_big_Omega*(sin_inc*sin_obliquity + cos_inc*cos_obliquity*sin_precession_plus_little_omega) + cos_obliquity*cos_precession_plus_little_omega*sin_big_Omega) + (sin_ta_minus_precession*(cos_inc*cos_precession_plus_little_omega*cos_big_Omega - sin_precession_plus_little_omega*sin_big_Omega))/(-1 + f2)**2 + (cos_ta_minus_precession*sin_obliquity*(-(cos_obliquity*cos_big_Omega*sin_inc) + sin_obliquity*(cos_inc*cos_big_Omega*sin_precession_plus_little_omega + cos_precession_plus_little_omega*sin_big_Omega)))/(-1 + f1)**2))/(r**2*(1 + e*cos_ta))
      p_coeffs%p_zz = ((cos_precession_plus_little_omega**2*sin_inc**2)/(-1 + f2)**2 + (cos_inc*sin_obliquity - cos_obliquity*sin_inc*sin_precession_plus_little_omega)**2 + (cos_inc*cos_obliquity + sin_inc*sin_obliquity*sin_precession_plus_little_omega)**2/(-1 + f1)**2)/r**2
      p_coeffs%p_z0 = (2*a*(-1 + e**2)*((cos_precession_plus_little_omega*sin_inc*sin_ta_minus_precession)/(-1 + f2)**2 + (cos_ta_minus_precession*(-((-2 + f1)*f1*cos_inc*cos_obliquity*sin_obliquity) + sin_inc*((-1 + f1)**2*cos_obliquity**2 + sin_obliquity**2)*sin_precession_plus_little_omega))/(-1 + f1)**2))/(r**2*(1 + e*cos_ta))
      p_coeffs%p_00 = (a**2*(-1 + e**2)**2*(sin_ta_minus_precession**2/(-1 + f2)**2 + (cos_ta_minus_precession**2*((-1 + f1)**2*cos_obliquity**2 + sin_obliquity**2))/(-1 + f1)**2))/(r + e*r*cos_ta)**2

   end subroutine compute_3d_coeffs

   subroutine compute_2d_coeffs(p_coeffs, rho_coeffs)
      implicit none
      type(p_coefficients), intent(in) :: p_coeffs
      type(rho_coefficients), intent(out) :: rho_coeffs

      rho_coeffs%rho_xx = p_coeffs%p_xx - p_coeffs%p_xz**2/(4.0*p_coeffs%p_zz)
      rho_coeffs%rho_xy = p_coeffs%p_xy - (p_coeffs%p_xz*p_coeffs%p_yz)/(2.0*p_coeffs%p_zz)
      rho_coeffs%rho_x0 = p_coeffs%p_x0 - (p_coeffs%p_xz*p_coeffs%p_z0)/(2.0*p_coeffs%p_zz)
      rho_coeffs%rho_yy = p_coeffs%p_yy - p_coeffs%p_yz**2/(4.0*p_coeffs%p_zz)
      rho_coeffs%rho_y0 = p_coeffs%p_y0 - (p_coeffs%p_yz*p_coeffs%p_z0)/(2.0*p_coeffs%p_zz)
      rho_coeffs%rho_00 = p_coeffs%p_00 - p_coeffs%p_z0**2/(4.0*p_coeffs%p_zz)

   end subroutine compute_2d_coeffs

end module three_d_coefficients
