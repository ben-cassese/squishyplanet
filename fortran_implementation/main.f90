program main
    !!!!!!!!!
   ! imports
    !!!!!!!!!
   use iso_fortran_env, only: dp => real64
   use model_types, only: orbit_parameters, planet_parameters_2d, planet_parameters_3d, p_coefficients
   use constants, only: PI
   use read_in_files, only: read_time_array, read_change_of_basis_matrix
   use keplerian, only: kepler, t0_to_t_peri
   use squishyplanet_2d, only: squishyplanet_lightcurve_2d
   use squishyplanet_3d, only: squishyplanet_lightcurve_3d
   implicit none

   real(dp), allocatable :: times(:)
   real(dp), allocatable :: fluxes(:)

   type(orbit_parameters) :: orbit_params
   type(planet_parameters_2d) :: planet_params
   type(planet_parameters_3d) :: planet_params_3d

   logical :: exists
   integer(8) :: count_rate, count_max, count_start, count_end
   real(8) :: elapsed_time

   ! this has to match the length of your ld_u_coeffs array
   ! will read in the appropriate change of basis matrix based on that
   real(dp), dimension(8) :: ld_u_coeffs

   character(len=256) :: filename
   real(dp), dimension(size(ld_u_coeffs) + 1, size(ld_u_coeffs) + 1) :: change_of_basis_matrix

    !!!!!!!!!!!
   ! setup
    !!!!!!!!!!!
   times = read_time_array('../times.txt') ! this allocates times
   allocate (fluxes(size(times)))

   write (filename, '(a,i0,a)') '../change_of_basis_matricies/g_matrix_', size(ld_u_coeffs), '.bin'
   change_of_basis_matrix = read_change_of_basis_matrix(filename, size(ld_u_coeffs))

   orbit_params%semi = 200.0_dp
   orbit_params%ecc = 0.3_dp
   orbit_params%inc = 89.75_dp*PI/180.0_dp
   orbit_params%big_Omega = 95.0_dp*PI/180.0_dp ! should always be pi/2 for transits, this is just to test it works in a full 3D scenario
   orbit_params%little_omega = PI/3.5_dp
   orbit_params%period = 1001.0_dp
   orbit_params%t0 = 0.2_dp

   planet_params%r_eff = 0.1_dp
   planet_params%f_squish_proj = 0.3_dp
   planet_params%theta_proj = 0.2_dp

   ld_u_coeffs = (/0.008_dp, 0.007_dp, 0.006_dp, 0.005_dp, 0.004_dp, 0.003_dp, 0.002_dp, 0.001_dp/)

   print *, "beginning the 2d version of squishyplanet"
   call SYSTEM_CLOCK(count_start, count_rate, count_max)
   call squishyplanet_lightcurve_2d( &
      ! these change with each sample
      orbit_params=orbit_params, &
      planet_params=planet_params, &
      ld_u_coeffs=ld_u_coeffs, &
      ! these don't change with each sample
      times=times, &
      fluxes=fluxes, &
      change_of_basis_matrix=change_of_basis_matrix &
      )
   call SYSTEM_CLOCK(count_end)
   elapsed_time = real(count_end - count_start) / REAL(count_rate)
   print *, 'Time taken = ', elapsed_time*1000.0_dp, ' ms'

   print *, "done, writing the lightcurve to 2d_lightcurve.bin"
   inquire(file='../2d_lightcurve.bin', exist=exists)
    if (exists) then
        open(unit=10, file='../lightcurve.bin')
        close(unit=10, status='delete')
    end if
   open(unit=10, file='../2d_lightcurve.bin', form='unformatted', access='stream', status='replace')
   write(10) fluxes
   close(10)


   print *, ""
   print *, "beginning the 3d version, different planet"

   orbit_params%semi = 200.0_dp
   orbit_params%ecc = 0.3_dp
   orbit_params%inc = 89.75_dp*PI/180.0_dp
   orbit_params%big_Omega = 95.0_dp*PI/180.0_dp
   orbit_params%little_omega = PI/3.5_dp
   orbit_params%period = 1001.0_dp
   orbit_params%t0 = 0.2_dp

   planet_params_3d%r = 0.2_dp
   planet_params_3d%f_squish_1 = 0.1_dp
   planet_params_3d%f_squish_2 = 0.2_dp
   planet_params_3d%obliquity = 0.3_dp
   planet_params_3d%precession = 0.4_dp

   call SYSTEM_CLOCK(count_start, count_rate, count_max)
   call squishyplanet_lightcurve_3d( &
      ! these change with each sample
      orbit_params=orbit_params, &
      planet_params=planet_params_3d, &
      ld_u_coeffs=ld_u_coeffs, &
      ! these don't change with each sample
      tidally_locked=.false., &
      times=times, &
      fluxes=fluxes, &
      change_of_basis_matrix=change_of_basis_matrix &
      )
   call SYSTEM_CLOCK(count_end)
   elapsed_time = real(count_end - count_start) / REAL(count_rate)
   print *, 'Time taken = ', elapsed_time*1000.0_dp, ' ms'

   print *, "done, writing the lightcurve to 3d_lightcurve.bin"
   inquire(file='../3d_lightcurve.bin', exist=exists)
   if (exists) then
      open(unit=10, file='../lightcurve.bin')
      close(unit=10, status='delete')
   end if
   open(unit=10, file='../3d_lightcurve.bin', form='unformatted', access='stream', status='replace')
   write(10) fluxes
   close(10)

   print *, ""
   print *, "now again, but tidally locked"
   call SYSTEM_CLOCK(count_start, count_rate, count_max)
   call squishyplanet_lightcurve_3d( &
      ! these change with each sample
      orbit_params=orbit_params, &
      planet_params=planet_params_3d, &
      ld_u_coeffs=ld_u_coeffs, &
      ! these don't change with each sample
      tidally_locked=.true., &
      times=times, &
      fluxes=fluxes, &
      change_of_basis_matrix=change_of_basis_matrix &
      )
   call SYSTEM_CLOCK(count_end)
   elapsed_time = real(count_end - count_start) / REAL(count_rate)
   print *, 'Time taken = ', elapsed_time*1000.0_dp, ' ms'

   print *, "done, writing the lightcurve to 3d_lightcurve_tl.bin"
   inquire(file='../3d_lightcurve_tl.bin', exist=exists)
   if (exists) then
      open(unit=10, file='../lightcurve.bin')
      close(unit=10, status='delete')
   end if
   open(unit=10, file='../3d_lightcurve_tl.bin', form='unformatted', access='stream', status='replace')
   write(10) fluxes
   close(10)

   deallocate (times)
   deallocate (fluxes)

end program main
