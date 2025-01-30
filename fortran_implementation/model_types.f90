module model_types
   use, intrinsic :: iso_fortran_env, only: dp => real64
   implicit none

   type :: orbit_parameters
      real(dp) :: semi
      real(dp) :: ecc
      real(dp) :: inc
      real(dp) :: big_Omega
      real(dp) :: little_omega
      real(dp) :: period
      real(dp) :: t0
   end type orbit_parameters


   type :: planet_parameters
      real(dp) :: r_eff
      real(dp) :: f_squish_proj
      real(dp) :: theta_proj
   end type planet_parameters


   type :: skypos_positions
      real(dp) :: x
      real(dp) :: y
      real(dp) :: z
   end type skypos_positions


   type :: rho_coefficients
      real(dp) :: rho_xx
      real(dp) :: rho_xy
      real(dp) :: rho_x0
      real(dp) :: rho_yy
      real(dp) :: rho_y0
      real(dp) :: rho_00
   end type rho_coefficients


   type :: para_coefficients
      real(dp) :: c_x1
      real(dp) :: c_x2
      real(dp) :: c_x3
      real(dp) :: c_y1
      real(dp) :: c_y2
      real(dp) :: c_y3
   end type para_coefficients


   type :: para_helper_coeffs
      real(dp) :: r1
      real(dp) :: r2
      real(dp) :: xc
      real(dp) :: yc
      real(dp) :: cosa
      real(dp) :: sina
   end type para_helper_coeffs

end module
