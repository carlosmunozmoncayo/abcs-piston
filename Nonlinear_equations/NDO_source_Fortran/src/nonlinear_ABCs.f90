!Source code to handle integral source term to damp
!the nonlinear characteristic variables in sponge layer
!q_t +f(q)_x = \int_{x}^{\infty} R(q)D(q)R(q)^{-1}q_x dx,
!where $D = diag(d1, d2, d3)*\Lambda$ and f'(q)=R\Lambda R^{-1}.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Assuming the solution is piecewise constant in each cell at every time step.
!The only contribution of the integral source term is at the cell interfaces.
!Meant to be used with operator splitting
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine Heun_step(q, n, gamma, dt, sigma_damping_vec, output)
    implicit none
    !Arguments
    integer, intent(in)  :: n
    double precision, intent(in) :: gamma, dt
    double precision, intent(in) :: q(3, n)
    double precision, intent(in) :: sigma_damping_vec(n)
    double precision, intent(out) :: output(3, n)
    
    !Local variables
    double precision :: q_tilde(3,n) !For time integration

    !Define interface for rhs
    interface
        function rhs(qi, n, gamma, sigma_damping_vec)
            implicit none
            !Arguments
            integer, intent(in)  :: n
            double precision, intent(in) :: gamma
            double precision, intent(in) :: qi(3, n)
            double precision, intent(in) :: sigma_damping_vec(n)
            double precision :: rhs(3, n)
        end function rhs
    end interface
    
    !Integrate q_t = rhs(q) with Heun's method
    q_tilde = q + dt*rhs(q, n, gamma, sigma_damping_vec)
    output = q + 0.5d0*dt*(rhs(q, n, gamma, sigma_damping_vec)+rhs(q_tilde, n, gamma, sigma_damping_vec))
end subroutine Heun_step

function construct_M(qi, gamma, sigma_damping)
    !Constructs the matrix M(qi)=R(qi)D(qi)R(qi)^{-1}
    implicit none
    !Arguments
    double precision :: gamma, sigma_damping
    double precision :: qi(3)
    double precision :: construct_M(3,3)
    !Local variables
    double precision :: V, u, eps, p
    double precision :: sv, sp, spv
    double precision :: lamb3, d1, d2, d3
    double precision :: A, B, C, J
    double precision :: sg

    V = qi(1)
    u = qi(2)
    eps = qi(3)
    p = (gamma-1.)*(eps-0.5*u**2)/V

    spv = dsqrt(p*V)
    sp = dsqrt(p)
    sv = dsqrt(V)
    sg = dsqrt(gamma)

    lamb3 = dsqrt(gamma)*sp/sv
    d3 = sigma_damping*lamb3
    d1 = 0.d0
    d2 = 0.d0

    A = d1-2.d0*d2+d3
    B = d1-d3
    C = gamma-1.d0
    J = 2.d0*sg*spv

    !First row
    construct_M(1,1) = (0.5/gamma)*(2*gamma*d2+A)
    construct_M(1,2) = (0.5/p/gamma)*(sg*spv*B+u*C*A)
    construct_M(1,3) = (0.5/p/gamma)*(-A*C)
    !Second row
    construct_M(2,1) = B*p/J
    construct_M(2,2) = (sg*spv*(d1+d3)+u*C*B)/J
    construct_M(2,3) = -B*C/J
    !Third ow
    construct_M(3,1) = (-p*V*A+B*spv*sg*u)/(2*V*gamma)
    construct_M(3,2) = (spv*u*A+sg*(C*u**2-p*V)*B)/(J*sg)
    construct_M(3,3) = (-spv*A-sg*C*B*u+gamma*spv*(d1+d3))/(J*sg)

end function construct_M

function rhs(q, n, gamma, sigma_damping_vec)
    implicit none
    !Arguments
    integer, intent(in)  :: n
    double precision, intent(in) :: gamma
    double precision, intent(in) :: q(3, n)
    double precision, intent(in) :: sigma_damping_vec(n)
    double precision :: rhs(3, n)
    !Local variables
    integer :: i
    double precision :: suma(3)
    double precision :: avg_q(3)

    !Define interface for construction of M(q)=R(q)D(q)R(q)^{-1}
    interface
        function construct_M(qi, gamma, sigma_damping)
            implicit none
            !Arguments
            double precision :: gamma, sigma_damping
            double precision :: qi(3)
            double precision :: construct_M(3,3)
        end function construct_M
    end interface

    !Initialize sum and output
    suma = 0.d0
    rhs = 0.d0

    do i = n-1, 1, -1
        if (sigma_damping_vec(i)<=1E-12) exit
        avg_q = 0.5d0*(q(:,i)+q(:,i+1))
        suma = suma + matmul(construct_M(avg_q, gamma, sigma_damping_vec(i)), (q(:,i+1)-q(:,i)))
        rhs(:,i) = suma
    end do
end function rhs
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Solution reconstructed as piecewise linear using the minmod slope limiter
!before computing source term.
!The idea is to take into account the contribution of the integral inside the cells.
!Meant to 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine rhs_2nd_order(q, n, gamma, sigma_damping_vec, x, output)
    !Each element of output contains an approximation of 
    !\int_{x_i}^{\infty} M(q)q_x dx
    implicit none
    !Arguments
    integer, intent(in)  :: n
    double precision, intent(in) :: gamma
    double precision, intent(in) :: q(3, n)
    double precision, intent(in) :: x(n)
    double precision, intent(in) :: sigma_damping_vec(n)
    double precision, intent(out) :: output(3, n)
    !double precision :: rhs(3, n)
    !Local variables
    integer :: i!For loops
    double precision :: suma(3)
    double precision :: avg_q(3)
    double precision :: slope_i(3), slope_ip1(3)
    double precision :: dx !Cell width
    double precision :: qSimpson(3,3) !Solution at Simpson points
    double precision :: avg_sigma_damping

    !Define interface block
    interface
        function construct_M(qi, gamma, sigma_damping)
            implicit none
            !Arguments
            double precision :: gamma, sigma_damping
            double precision :: qi(3)
            double precision :: construct_M(3,3)
        end function construct_M

        function eval_reconst_func(qi,xi,slopes,x)
            implicit none
            !Arguments
            double precision, intent(in) :: qi(3), slopes(3), xi, x
            double precision :: eval_reconst_func(3)
        end function eval_reconst_func

        function minmod_slope(q_slice, dx)
            implicit none
            !Arguments
            double precision, intent(in) :: q_slice(3,3), dx
            double precision :: minmod_slope(3)
        end function minmod_slope
    end interface

    !Initialize sum ,output, and solution at Simpson nodes
    qSimpson = 0.d0

    suma = 0.d0
    output = 0.d0

    dx = x(2)-x(1)

    do i = n-2, 2, -1 !Skipping first and last cells (Nothing sholud happen there)
        if (sigma_damping_vec(i)<=1E-12) exit
        !!!!!!!!!!!
        !Integral from x_i to x_{i+1/2}
        !!!!!!!!!!!
        slope_i = minmod_slope(q(:,i-1:i+1), dx)
        !Get Simpson nodes
        qSimpson(:,1) = eval_reconst_func(q(:,i), x(i), slope_i, x(i)) !a
        qSimpson(:,2) = eval_reconst_func(q(:,i), x(i), slope_i, x(i)+dx/4.d0) !a+quad_dx/2
        qSimpson(:,3) = eval_reconst_func(q(:,i), x(i), slope_i, x(i)+dx/2.d0) !b
        !Compute quadrature
        suma = suma + (dx/12.d0)*matmul(construct_M(qSimpson(:,1), gamma, sigma_damping_vec(i)), slope_i)
        suma = suma + (dx/3.d0)*matmul(construct_M(qSimpson(:,2), gamma, sigma_damping_vec(i)), slope_i)
        suma = suma + (dx/12.d0)*matmul(construct_M(qSimpson(:,3), gamma, sigma_damping_vec(i)), slope_i)


        !Integral from x_{i+1/2} to x_{i+1}
        slope_ip1 = minmod_slope(q(:,i:i+2), dx)
        !Get Simpson nodes
        qSimpson(:,1) = eval_reconst_func(q(:,i+1), x(i+1), slope_ip1, x(i+1)-dx/2.d0) !a
        qSimpson(:,2) = eval_reconst_func(q(:,i+1), x(i+1), slope_ip1, x(i+1)-dx/4.d0) !a+quad_dx/2
        qSimpson(:,3) = eval_reconst_func(q(:,i+1), x(i+1), slope_ip1, x(i+1)) !b
        !Compute quadrature
        suma = suma + (dx/12.d0)*matmul(construct_M(qSimpson(:,1), gamma, sigma_damping_vec(i+1)), slope_ip1)
        suma = suma + (dx/3.d0)*matmul(construct_M(qSimpson(:,2), gamma, sigma_damping_vec(i+1)), slope_ip1)
        suma = suma + (dx/12.d0)*matmul(construct_M(qSimpson(:,3), gamma, sigma_damping_vec(i+1)), slope_ip1)

        !Boundary integral from x_{i+1/2}^- to x_{i+1/2}^+
        avg_q = 0.5d0*(q(:,i)+q(:,i+1))
        avg_sigma_damping = 0.5d0*(sigma_damping_vec(i)+sigma_damping_vec(i+1))
        suma = suma + matmul(construct_M(avg_q, gamma, avg_sigma_damping), (q(:,i+1)-q(:,i)))
        output(:,i) = suma
    end do
end subroutine rhs_2nd_order

function eval_reconst_func(qi,xi,slopes,x)
    implicit none
    !Arguments
    double precision, intent(in) :: qi(3), slopes(3), x, xi
    double precision :: eval_reconst_func(3)
    integer :: i

    do i=1, 3
        eval_reconst_func(i) = qi(i) + slopes(i)*(x-xi)
    end do
end function eval_reconst_func

function minmod_slope(q_slice, dx)
    implicit none
    !Arguments
    double precision, intent(in) :: q_slice(3,3), dx
    double precision :: minmod_slope(3)
    !Local variables
    double precision :: q_im1(3), q_i(3), q_ip1(3)
    integer :: k

    !Define interface minmod
    interface
        function minmod(a,b)
            implicit none
            !Arguments
            double precision, intent(in) :: a, b
            double precision :: minmod
        end function minmod
    end interface

    q_im1 = q_slice(:,1)
    q_i = q_slice(:,2)
    q_ip1 = q_slice(:,3)

    do k = 1, 3
        minmod_slope(k) = minmod((1.d0/dx)*(q_ip1(k)-q_i(k)), (1.d0/dx)*(q_i(k)-q_im1(k)))
    end do
end function minmod_slope

function minmod(a,b)
    implicit none
    !Arguments
    double precision, intent(in) :: a, b
    double precision :: minmod

    if (a*b>0.d0) then
        if (abs(a)<abs(b)) then
            minmod = a
        else
            minmod = b
        end if
    else
        minmod = 0.d0
    end if
end function minmod
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    






