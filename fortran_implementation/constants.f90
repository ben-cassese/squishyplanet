module constants
   use, intrinsic :: iso_fortran_env, only: dp => real64
   implicit none
   real(dp), parameter :: PI = 3.14159265358979323846_dp
   real(dp), parameter :: TWO_PI = 2.0_dp*PI
   real(dp), parameter :: HALF_PI = 0.5_dp*PI
   real(dp), parameter :: THREE_HALF_PI = 1.5_dp*PI
   real(dp), parameter :: DEG_TO_RAD = PI/180.0_dp
   real(dp), parameter :: RAD_TO_DEG = 180.0_dp/PI
end module constants
