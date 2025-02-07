module squishyplanet_luna_interface
   use iso_fortran_env, only: dp => real64
   use read_in_files, only: read_change_of_basis_matrix
   use model_types, only: orbit_parameters, planet_parameters_2d, planet_parameters_3d
   use constants, only: PI
   use keplerian, only: kepler, t0_to_t_peri
   use squishyplanet_2d, only: squishyplanet_lightcurve_2d
   use squishyplanet_3d, only: squishyplanet_lightcurve_3d
   implicit none
   PRIVATE

   public :: read_change_of_basis_matrix
   public :: plan_squishyplanet_2d
   public :: plan_squishyplanet_3d

contains

! luna-type call
! call plan(times,Pdays,0.0D0,p,aR,e,wrad,bimpact,&
! u1,u2,fpri,ndata,fluxes)

   subroutine plan_squishyplanet_2d( &
      times, &
      Pdays, &
      unused1, &
      p, &
      aR, &
      e, &
      wrad, &
      bimpact, &
      u1, &
      u2, &
      unused2, &
      t0, &
      f_squish_proj, &
      theta_proj, &
      change_of_basis_matrix, &
      unused3, &
      fluxes &
      )

      real(dp), intent(in) :: times(:)
      real(dp), intent(in) :: Pdays, unused1, p, aR, e, wrad, bimpact, u1, u2, unused2, t0, f_squish_proj, theta_proj
      real(dp), intent(in) :: change_of_basis_matrix(:, :)
      integer, intent(in) :: unused3
      real(dp), intent(out) :: fluxes(size(times))

      type(orbit_parameters) :: orbit_params
      type(planet_parameters_2d) :: planet_params
      real(dp) :: ld_u_coeffs(2)

      orbit_params%semi = aR
      orbit_params%ecc = e
      orbit_params%inc = dacos(bimpact/aR)
      orbit_params%big_Omega = PI
      orbit_params%little_omega = wrad
      orbit_params%period = Pdays
      orbit_params%t0 = t0

      planet_params%r_eff = p
      planet_params%f_squish_proj = f_squish_proj
      planet_params%theta_proj = theta_proj

      ld_u_coeffs = (/u1, u2/)

      call squishyplanet_lightcurve_2d( &
         orbit_params=orbit_params, &
         planet_params=planet_params, &
         ld_u_coeffs=ld_u_coeffs, &
         times=times, &
         change_of_basis_matrix=change_of_basis_matrix, &
         fluxes=fluxes &
         )
   end subroutine plan_squishyplanet_2d

   subroutine plan_squishyplanet_3d( &
      times, &
      Pdays, &
      unused1, &
      p, &
      aR, &
      e, &
      wrad, &
      bimpact, &
      u1, &
      u2, &
      unused2, &
      t0, &
      f1, &
      f2, &
      obliquity, &
      precession, &
      tidally_locked, &
      change_of_basis_matrix, &
      unused3, &
      fluxes &
      )

      real(dp), intent(in) :: times(:)
      real(dp), intent(in) :: Pdays, unused1, p, aR, e, wrad, bimpact, u1, u2, unused2, t0
      real(dp), intent(in) :: f1, f2, obliquity, precession
      logical, intent(in) :: tidally_locked
      real(dp), intent(in) :: change_of_basis_matrix(:, :)
      integer, intent(in) :: unused3
      real(dp), intent(out) :: fluxes(size(times))

      type(orbit_parameters) :: orbit_params
      type(planet_parameters_3d) :: planet_params
      real(dp) :: ld_u_coeffs(2)

      orbit_params%semi = aR
      orbit_params%ecc = e
      orbit_params%inc = dacos(bimpact/aR)
      orbit_params%big_Omega = PI
      orbit_params%little_omega = wrad
      orbit_params%period = Pdays
      orbit_params%t0 = t0

      planet_params%r = p
      planet_params%f_squish_1 = f1
      planet_params%f_squish_2 = f2
      planet_params%obliquity = obliquity
      planet_params%precession = precession

      ld_u_coeffs = (/u1, u2/)

      call squishyplanet_lightcurve_3d( &
         orbit_params=orbit_params, &
         planet_params=planet_params, &
         ld_u_coeffs=ld_u_coeffs, &
         tidally_locked=tidally_locked, &
         times=times, &
         change_of_basis_matrix=change_of_basis_matrix, &
         fluxes=fluxes &
         )

   end subroutine plan_squishyplanet_3d

end module squishyplanet_luna_interface
