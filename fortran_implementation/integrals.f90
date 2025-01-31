module integral_helpers
   use iso_fortran_env, only: dp => real64
   use model_types, only: para_coefficients
   implicit none
   integer :: s_n
   type(para_coefficients) :: para_helper
end module integral_helpers

module solution_vecs
   use iso_fortran_env, only: dp => real64
   use constants, only: HALF_PI, PI, THREE_HALF_PI, TWO_PI
   use quadpack, only: dqag
   use model_types, only: para_coefficients
   implicit none

   real(dp), parameter :: epsabs = 1.0e-12_dp ! Absolute error tolerance
   real(dp), parameter :: epsrel = 1.0e-12_dp ! Relative error tolerance
   integer, parameter :: key = 5 ! selects Gauss-Kronrod 25,51 points
   real(dp) :: result ! the estimated integral
   real(dp) :: abserr ! the estimated absolute error
   integer :: neval ! the number of function evaluations
   integer :: ier ! an error flag
   integer, parameter :: limit = 100 ! maximum number of subintervals
   integer, parameter :: lenw = limit*4 ! need to store endpoints, val, err for each subinterval
   integer :: last ! the number of subintervals actually used
   integer :: iwork(limit) ! for parsing the work array for dqag
   real(dp) :: work(lenw) ! work array for dqag

contains

   subroutine star_solution_vec(a, b, g_coeffs, para, solution_vector)
      implicit none

      real(dp), intent(in) :: a, b
      real(dp), intent(in), allocatable :: g_coeffs(:)
      type(para_coefficients), intent(in) :: para
      real(dp), intent(out), allocatable :: solution_vector(:)

      ! local variables
      real(dp) :: x1, y1, x2, y2, theta1, theta2, theta1_tmp, theta2_tmp, delta
      logical :: wrap

      solution_vector = g_coeffs*0.0_dp

      x1 = para%c_x1*dcos(a) + para%c_x2*dsin(a) + para%c_x3
      y1 = para%c_y1*dcos(a) + para%c_y2*dsin(a) + para%c_y3
      theta1_tmp = datan2(y1, x1)
      if (theta1_tmp < 0.0_dp) then
         theta1_tmp = theta1_tmp + 2.0_dp*PI
      end if

      x2 = para%c_x1*dcos(b) + para%c_x2*dsin(b) + para%c_x3
      y2 = para%c_y1*dcos(b) + para%c_y2*dsin(b) + para%c_y3
      theta2_tmp = datan2(y2, x2)
      if (theta2_tmp < 0.0_dp) then
         theta2_tmp = theta2_tmp + 2.0_dp*PI
      end if

      ! Order the angles
      if (theta1_tmp < theta2_tmp) then
         theta1 = theta1_tmp
         theta2 = theta2_tmp
      else
         theta1 = theta2_tmp
         theta2 = theta1_tmp
      end if

      delta = theta2 - theta1

      wrap = delta > PI
      ! print *, "delta: ", delta

      if (wrap .neqv. .true.) then
         ! print *, "no wrap integration"
         call dqag(s0_integrand_star, theta1, theta2, epsabs, epsrel, key, result, &
                   abserr, neval, ier, limit, lenw, last, &
                   iwork, work)
         solution_vector(1) = result

         call dqag(s1_integrand_star, theta1, theta2, epsabs, epsrel, key, result, &
                   abserr, neval, ier, limit, lenw, last, &
                   iwork, work)
         solution_vector(2) = result

         return
      else
         ! print *, "wrap integration"
         call dqag(s0_integrand_star, theta2, TWO_PI, epsabs, epsrel, key, result, &
                   abserr, neval, ier, limit, lenw, last, &
                   iwork, work)
         solution_vector(1) = result

         call dqag(s0_integrand_star, 0.0_dp, theta1, epsabs, epsrel, key, result, &
                   abserr, neval, ier, limit, lenw, last, &
                   iwork, work)
         solution_vector(1) = solution_vector(1) + result

         call dqag(s1_integrand_star, theta2, TWO_PI, epsabs, epsrel, key, result, &
                   abserr, neval, ier, limit, lenw, last, &
                   iwork, work)
         solution_vector(2) = result

         call dqag(s1_integrand_star, 0.0_dp, theta1, epsabs, epsrel, key, result, &
                   abserr, neval, ier, limit, lenw, last, &
                   iwork, work)
         solution_vector(2) = solution_vector(2) + result
      end if

   contains
      function s0_integrand_star(t) result(val)
         implicit none
         real(dp), intent(in) :: t
         real(dp) :: val
         real(dp) :: tmp
         tmp = dcos(t)
         val = tmp*tmp
      end function s0_integrand_star

      function s1_integrand_star(t) result(val)
         use constants, only: PI, HALF_PI, THREE_HALF_PI
         implicit none
         real(dp), intent(in) :: t
         real(dp) :: val

         real(dp) :: cos_t, cos_2t
         cos_t = dcos(t)
         cos_2t = dcos(2.0_dp*t)

         if (t < HALF_PI .or. t > THREE_HALF_PI) then
            val = (PI*cos_t*(5.0_dp + 3.0_dp*cos_2t))/24.0_dp
         else
            val = -(PI*cos_t*(1.0_dp + 3.0_dp*cos_2t))/24.0_dp
         end if
      end function s1_integrand_star

   end subroutine star_solution_vec

   subroutine planet_solution_vec(a, b, g_coeffs, para, solution_vector)
      use integral_helpers
      implicit none

      real(dp), intent(in) :: a, b
      real(dp), intent(in), allocatable :: g_coeffs(:)
      type(para_coefficients), intent(in) :: para
      real(dp), intent(out), allocatable :: solution_vector(:)

      integer :: i

      ! copy para into the shared module so
      ! the integrand functions can access it
      para_helper = para

      solution_vector = g_coeffs*0.0_dp

      call dqag(s0_integrand_planet, a, b, epsabs, epsrel, key, result, &
                abserr, neval, ier, limit, lenw, last, &
                iwork, work)
      solution_vector(1) = result

      call dqag(s1_integrand_planet, a, b, epsabs, epsrel, key, result, &
                abserr, neval, ier, limit, lenw, last, &
                iwork, work)
      solution_vector(2) = result

      s_n = 2
      do i = s_n, size(g_coeffs)
         call dqag(sn_integrand_planet, a, b, epsabs, epsrel, key, result, &
                   abserr, neval, ier, limit, lenw, last, &
                   iwork, work)
         solution_vector(i + 1) = result
         s_n = s_n + 1
      end do

   contains
      function s0_integrand_planet(t) result(val)
         use integral_helpers
         implicit none
         real(dp), intent(in) :: t
         real(dp) :: val

         real(dp) :: cos_t, sin_t
         cos_t = dcos(t)
         sin_t = dsin(t)
         val = (cos_t*para_helper%c_x1 + sin_t*para_helper%c_x2 + para_helper%c_x3)* &
               (-sin_t*para_helper%c_y1 + cos_t*para_helper%c_y2)

      end function s0_integrand_planet

      function s1_integrand_planet(t) result(val)
         use integral_helpers
         implicit none
         real(dp), intent(in) :: t
         real(dp) :: val

         real(dp) :: x_term, y_term, sqrt_term, atan_term

         x_term = dcos(t)*para_helper%c_x1 + dsin(t)*para_helper%c_x2 + para_helper%c_x3

         y_term = dcos(t)*para_helper%c_y1 + dsin(t)*para_helper%c_y2 + para_helper%c_y3

         sqrt_term = dsqrt(1.0_dp - x_term**2 - y_term**2)

         atan_term = datan(x_term/sqrt_term)

         val = (-(dsin(t)*para_helper%c_y1) + dcos(t)*para_helper%c_y2)*( &
               PI + &
               6.0_dp*x_term*sqrt_term - &
               6.0_dp*atan_term*(-1.0_dp + y_term**2) &
               )/12.0_dp

      end function s1_integrand_planet

      function sn_integrand_planet(t) result(val)
         use integral_helpers
         implicit none

         real(dp), intent(in) :: t
         real(dp) :: val

         real(dp) :: cos_s, sin_s, tmp1, tmp2, squared_terms, power_term, cross_products

         cos_s = cos(t)
         sin_s = sin(t)

         tmp1 = cos_s*para_helper%c_x1 + sin_s*para_helper%c_x2 + para_helper%c_x3
         tmp2 = cos_s*para_helper%c_y1 + sin_s*para_helper%c_y2 + para_helper%c_y3

         squared_terms = 1.0d0 - tmp1*tmp1 - tmp2*tmp2

         ! s_n comes from the shared module integral_helpers
         power_term = squared_terms**(s_n/2.0d0)

         cross_products = para_helper%c_x3*(sin_s*para_helper%c_y1 - cos_s*para_helper%c_y2) + &
                          para_helper%c_x2*(para_helper%c_y1 + cos_s*para_helper%c_y3) - &
                          para_helper%c_x1*(para_helper%c_y2 + sin_s*para_helper%c_y3)

         val = -(power_term*cross_products)

      end function sn_integrand_planet

   end subroutine planet_solution_vec

end module solution_vecs
