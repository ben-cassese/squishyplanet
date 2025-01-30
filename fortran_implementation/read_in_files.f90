module read_in_files
   use, intrinsic :: iso_fortran_env, only: dp => real64
   implicit none

contains

   function read_time_array(filename) result(times)
      character(len=*), intent(in) :: filename
      real(dp), allocatable :: times(:)
      integer :: io_unit, n_times, i, io_status

      ! Open file
      open (newunit=io_unit, file=filename, status='old', action='read')

      ! Count lines
      n_times = 0
      do
         read (io_unit, *, iostat=io_status)
         if (io_status /= 0) exit  ! Exit if we hit end of file or error
         n_times = n_times + 1
      end do

      ! Go back to start
      rewind (io_unit)

      ! Read values
      allocate (times(n_times))
      do i = 1, n_times
         read (io_unit, *) times(i)
      end do

      close (io_unit)
   end function


   function read_change_of_basis_matrix(filename, dims) result(matrix)
      character(len=*), intent(in) :: filename
      integer, intent(in) :: dims
      real(dp), dimension((dims+1),(dims+1)) :: matrix
      integer :: file_unit

      open(newunit=file_unit, file=filename, form='unformatted', access='stream', status='old')

      read(file_unit) matrix

      close(file_unit)

  end function

end module read_in_files
