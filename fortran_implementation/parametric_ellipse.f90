module parametric_ellipse
   use, intrinsic :: iso_fortran_env, only: dp => real64
   use model_types, only: rho_coefficients, para_helper_coeffs, para_coefficients
   use constants, only: PI, HALF_PI

contains

   function calculate_rho_coefficients(projected_r, projected_f, projected_theta, &
                                       xc, yc) result(rho_coeffs)
      ! same as the rho calcs in parameterize_2d_helper in sq.engine.polynomial_limb_darkened_transit
      ! but, even with identical inputs/structure, ~1e-14 differences between the two
      ! running w/ it
      implicit none

      ! Input parameters
      real(dp), intent(in) :: projected_r     ! Input radius
      real(dp), intent(in) :: projected_f     ! Input f parameter
      real(dp), intent(in) :: projected_theta ! Input theta angle
      real(dp), intent(in) :: xc, yc         ! Input center coordinates

      ! Output structure
      type(rho_coefficients) :: rho_coeffs

      ! Local variables
      real(dp) :: cos_t, sin_t, projected_r2
      real(dp) :: projected_r_sq, projected_r2_sq, cos_t_sq, sin_t_sq, xc_sq, yc_sq
      real(dp) :: tmp

      ! Calculate intermediate values
      projected_r2 = projected_r*(1.0_dp - projected_f)
      cos_t = cos(projected_theta)
      sin_t = sin(projected_theta)

      projected_r_sq = projected_r*projected_r
      projected_r2_sq = projected_r2*projected_r2
      cos_t_sq = cos_t*cos_t
      sin_t_sq = sin_t*sin_t
      xc_sq = xc*xc
      yc_sq = yc*yc

      rho_coeffs%rho_xx = cos_t_sq/projected_r_sq + &
                          sin_t_sq/projected_r2_sq

      tmp = (2.0_dp*cos_t*sin_t)
      rho_coeffs%rho_xy = tmp/projected_r_sq - &
                          tmp/projected_r2_sq


      tmp = 2.0_dp*cos_t*yc*sin_t
      rho_coeffs%rho_x0 = ((-2.0_dp*cos_t_sq*xc) - tmp)/projected_r_sq &
                          + (tmp - (2.0_dp*xc*sin_t_sq))/projected_r2_sq

      rho_coeffs%rho_yy = cos_t_sq/projected_r2_sq + &
                          sin_t_sq/projected_r_sq

      tmp = 2.0_dp*cos_t*xc*sin_t
      rho_coeffs%rho_y0 = ((-2.0_dp*cos_t_sq*yc) + (tmp))/projected_r2_sq &
                          - ((2.0_dp*cos_t*xc*sin_t) + (2.0_dp*yc*sin_t_sq))/projected_r_sq

      tmp = 2.0_dp*cos_t*xc*yc*sin_t
      rho_coeffs%rho_00 = ((cos_t_sq*xc_sq) + (tmp) + (yc_sq*sin_t_sq))/projected_r_sq + &
                          ((cos_t_sq*yc_sq) - (tmp) + (xc_sq*sin_t_sq))/projected_r2_sq

   end function calculate_rho_coefficients

   function poly_to_parametric_helper(rho) &
      result(para_helpers)
      implicit none

      ! Input parameters
      type(rho_coefficients), intent(in) :: rho

      ! Output parameters
      type(para_helper_coeffs) :: para_helpers

      ! Local variables
      real(dp) :: rho_xx_shift, rho_xy_shift, rho_yy_shift
      real(dp) :: theta, denom, a, b
      real(dp) :: diff_xx_yy

      ! Calculate the center of the ellipse
      denom = 4.0_dp*rho%rho_xx*rho%rho_yy - rho%rho_xy**2
      para_helpers%xc = (rho%rho_xy*rho%rho_y0 - 2.0_dp*rho%rho_yy*rho%rho_x0)/denom
      para_helpers%yc = (rho%rho_xy*rho%rho_x0 - 2.0_dp*rho%rho_xx*rho%rho_y0)/denom

      ! Calculate shifted coefficients for centered ellipse
      denom = (-1.0_dp + rho%rho_00)*rho%rho_xy**2 - rho%rho_x0*rho%rho_xy*rho%rho_y0 + &
              rho%rho_x0**2*rho%rho_yy + rho%rho_xx*(rho%rho_y0**2 + 4.0_dp*rho%rho_yy - &
                                                     4.0_dp*rho%rho_00*rho%rho_yy)

      rho_xx_shift = -(rho%rho_xx*(rho%rho_xy**2 - 4.0_dp*rho%rho_xx*rho%rho_yy))/denom
      rho_xy_shift = (-(rho%rho_xy**3) + 4.0_dp*rho%rho_xx*rho%rho_xy*rho%rho_yy)/denom
      rho_yy_shift = -(rho%rho_yy*(rho%rho_xy**2 - 4.0_dp*rho%rho_xx*rho%rho_yy))/denom

      ! Calculate rotation angle
      diff_xx_yy = rho_xx_shift - rho_yy_shift
      if (abs(diff_xx_yy) > tiny(1.0_dp)) then
         theta = 0.5_dp*atan2(rho_xy_shift, diff_xx_yy) + HALF_PI
         if (theta < 0.0_dp) then
            theta = theta + PI
         end if
      else
         theta = 0.0_dp
      end if

      ! Calculate sine and cosine of rotation angle
      para_helpers%cosa = cos(theta)
      para_helpers%sina = sin(theta)

      ! Calculate semi-major and semi-minor axes
      a = rho_xx_shift*para_helpers%cosa**2 + rho_xy_shift*para_helpers%cosa*para_helpers%sina + &
          rho_yy_shift*para_helpers%sina**2
      b = rho_xx_shift*para_helpers%sina**2 - rho_xy_shift*para_helpers%cosa*para_helpers%sina + &
          rho_yy_shift*para_helpers%cosa**2

      ! Calculate final radii
      para_helpers%r1 = 1.0_dp/sqrt(a)
      para_helpers%r2 = 1.0_dp/sqrt(b)

   end function poly_to_parametric_helper

   function poly_to_parametric(rho) result(para)
      implicit none

      ! Input parameters
      type(rho_coefficients), intent(in) :: rho

      ! Output parameters
      type(para_coefficients) :: para

      ! Local variables
      type(para_helper_coeffs) :: para_helpers

      ! Calculate helper coefficients
      para_helpers = poly_to_parametric_helper(rho)

      ! Calculate final coefficients
      para%c_x1 = para_helpers%r1*para_helpers%cosa
      para%c_x2 = -para_helpers%r2*para_helpers%sina
      para%c_x3 = para_helpers%xc
      para%c_y1 = para_helpers%r1*para_helpers%sina
      para%c_y2 = para_helpers%r2*para_helpers%cosa
      para%c_y3 = para_helpers%yc

   end function poly_to_parametric


   function cartesian_intersection_to_parametric_angle(xs, ys, para) result(alphas)
      implicit none

      real(dp), intent(in) :: xs(4)
      real(dp), intent(in) :: ys(4)
      type(para_coefficients), intent(in) :: para
      real(dp) :: alphas(4)

      ! Local variables
      real(dp) :: det, xs_centered(4), ys_centered(4), cosa(4), sina(4)
      real(dp) :: inv_matrix(2,2)
      integer :: i

      ! Center the ellipse by subtracting c_x3 and c_y3
      xs_centered = xs - para%c_x3
      ys_centered = ys - para%c_y3

      ! Calculate inverse matrix
      det = para%c_x1 * para%c_y2 - para%c_x2 * para%c_y1
      inv_matrix(1,1) = para%c_y2 / det
      inv_matrix(1,2) = -para%c_x2 / det
      inv_matrix(2,1) = -para%c_y1 / det
      inv_matrix(2,2) = para%c_x1 / det

      ! Calculate cosa and sina for each point
      do i = 1, 4
         cosa(i) = inv_matrix(1,1) * xs_centered(i) + inv_matrix(1,2) * ys_centered(i)
         sina(i) = inv_matrix(2,1) * xs_centered(i) + inv_matrix(2,2) * ys_centered(i)
      end do

      ! Calculate alpha using atan2
      do i = 1, 4
         alphas(i) = datan2(sina(i), cosa(i))
      end do

   end function cartesian_intersection_to_parametric_angle

end module parametric_ellipse
