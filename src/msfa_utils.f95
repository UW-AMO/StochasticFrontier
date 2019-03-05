! Fortran module for the special function calculation
module special

contains

! -----------------------------------------------------------------------------
function rlog_1p_erf(n, x) result(y)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: x(:)
    integer :: i
    real(8) :: spi = sqrt(4.d0*atan(1.d0))
    real(8) :: y(n)

    do i = 1, n
        if (x(i) < -5.d0) then
            y(i) = -x(i)**2 - log(-spi*x(i))
        else
            y(i) = log(1.d0 + erf(x(i)))
        endif
    enddo
end function rlog_1p_erf

function clog_1p_erf(n, x) result(y)
    implicit none
    integer, intent(in) :: n
    complex(8), intent(in) :: x(:)
    integer :: i
    real(8) :: spi = sqrt(4.d0*atan(1.d0))
    real(8) :: r, c
    complex(8) :: y(n)

    do i = 1, n
        r = real(x(i))
        c = imag(x(i))
        if (r < -5.d0) then
            y(i) = cmplx(-r**2 - log(-spi*r), c*(-2.d0*r - 1.d0/r))
        else
            y(i) = cmplx(log(1.d0 + erf(r)), &
                c*2.d0*exp(-r**2)/(spi*(1.d0 + erf(r))))
        endif
    enddo
end function clog_1p_erf

! -----------------------------------------------------------------------------
function rlog_1m_erf(n, x) result(y)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: x(:)
    integer :: i
    real(8) :: spi = sqrt(4.d0*atan(1.d0))
    real(8) :: y(n)

    do i = 1, n
        if (x(i) > 5.d0) then
            y(i) = -x(i)**2 - log(spi*x(i))
        else
            y(i) = log(1.d0 - erf(x(i)))
        endif
    enddo
end function rlog_1m_erf

function clog_1m_erf(n, x) result(y)
    implicit none
    integer, intent(in) :: n
    complex(8), intent(in) :: x(:)
    integer :: i
    real(8) :: spi = sqrt(4.d0*atan(1.d0))
    real(8) :: r, c
    complex(8) :: y(n)

    do i = 1, n
        r = real(x(i))
        c = imag(x(i))
        if (r > 5.d0) then
            y(i) = cmplx(-r**2 - log(spi*r), c*(-2.d0*r - 1.d0/r))
        else
            y(i) = cmplx(log(1.d0 - erf(r)), &
                -c*2.d0*exp(-r**2)/(spi*(1.d0 - erf(r))))
        endif
    enddo
end function clog_1m_erf

end module special