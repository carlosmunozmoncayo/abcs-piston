#!/usr/bin/env python
# encoding: utf-8
r"""
Piston problem in Lagrangian coordinates
===================================

Solve the one-dimensional Euler equations for inviscid, compressible flow
writen in Lagrangian coordinates:

.. math::
\frac{D}{D t}\left(\begin{array}{l}
V \\
v \\
p
\end{array}\right)
+
\left(\begin{array}{ccc}
0 & -1 & 0 \\
0 & 0 & 1 \\
0 & d^2 & 0
\end{array}\right) \frac{\partial}{\partial \xi}\left(\begin{array}{l}
V \\
v \\
p
\end{array}\right)
=0

The fluid is an ideal gas, with pressure given by :math:`p=\rho (\gamma-1)e - gamma p_{\inf}`
where e is internal energy and p+{\inf} represents the cohesion effects in liquid and solid states.

xi is a lagrangian coordinate and D/Dt is the material derivative.

The objective of this script is to solve the Euler equations in Lagrangian coordinates,
where the left boundary condition is prescribed by a moving piston.

"""
from __future__ import absolute_import
from __future__ import print_function
from clawpack import riemann
from clawpack import pyclaw
import numpy as np
import sys
sys.path.append('./Riemann_solvers/modules')
#Importing Riemann solvers
import euler_HLL_1D
import euler_HLL_slowing_1D

#Importing clawpack source terms
from source_terms import clawpack_source_step_relaxation_scalar, clawpack_source_step_relaxation_matrix
from source_terms import clawpack_source_step_far_field

from source_terms import clawpack_source_step_far_field_scalar, clawpack_RK23_source_step_far_field


#Import sharpclaw source term
from source_terms import sharpclaw_source_step_damping, sharpclaw_source_step_damping_scalar
from source_terms import sharpclaw_source_step_nonlinear_far_field

#Import custom solver
from custom_solver_class import customSharpClawSolver1D as custom_solver

gamma = 1.4 # Ratio of specific heats

def setup(use_petsc=False,outdir='./_output',solver_type='sharpclaw',
        kernel_language='Fortran',time_integrator='Heun',mx=10000, tfinal= 40*np.pi,
        nout = 10, xmax = 30*2*np.pi, M = 0.4, CFL = 0.8, limiting = 1, order = 2, piston_freq=1., 
        euler_RS = 'euler',
        far_field_damping = False, far_field_damping_rate=10, 
        relaxation_method = False, matrix_filter=False,
        start_slowing = 1e+4, stop_slowing=1e+5, 
        start_absorbing=1e+4, stop_absorbing=1e+5,
        sigma_damping_type="affine", sigma_slowing_type="affine",
        scalar_far_field = False,
        implicit_integrator = False,
        integral_source=False,
        b_Mayer_filter=0.5,
        before_stage=False,
        adaptiveRM=False,
        Strang=False,
        constant_extrap=False):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if kernel_language =='Python':
        raise Exception('Python kernel not implemented for this problem')
    elif kernel_language =='Fortran':
        if euler_RS == 'euler_slowing':
            rs = euler_HLL_slowing_1D
        elif euler_RS == 'euler':
            rs = euler_HLL_1D
    else:
        raise Exception('Unrecognized kernel_language specified')

    if solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(rs)
        solver.lim_type = limiting
        solver.cfl_max = 0.9
        if time_integrator == 'Heun':
            a = np.array([[0.,0.],[1.,0.]])
            b = np.array([0.5,0.5])
            c = np.array([0.,1.])
            solver.time_integrator = 'RK'
            solver.a, solver.b, solver.c = a, b, c
        solver.cfl_desired = CFL
        solver.limiters = 1# minmod limiter pyclaw.limiters.tvd.minmod
        solver.max_steps = 5000000
        if far_field_damping:
            if scalar_far_field:
                solver.dq_src = sharpclaw_source_step_damping_scalar
            elif integral_source:
                solver.dq_src = sharpclaw_source_step_nonlinear_far_field
            else:
                solver.dq_src = sharpclaw_source_step_damping
        solver.call_before_step_each_stage = before_stage


    elif solver_type == 'sharpclaw_custom':
        Strang = True
        solver = custom_solver(riemann_solver=rs,claw_package='clawpack.pyclaw',
                far_field_damping = far_field_damping, 
                relaxation_method = relaxation_method, 
                matrix_filter = matrix_filter,
                scalar_far_field=scalar_far_field,
                implicit_integrator = implicit_integrator,
                integral_source=integral_source,
                adaptiveRM=adaptiveRM,
                Strang=Strang)

        solver.lim_type = limiting
        solver.cfl_max = 0.9
        if time_integrator == 'Heun':
            a = np.array([[0.,0.],[1.,0.]])
            b = np.array([0.5,0.5])
            c = np.array([0.,1.])
            solver.time_integrator = 'RK'
            solver.a, solver.b, solver.c = a, b, c
        solver.cfl_desired = CFL
        solver.limiters = 1# minmod limiter pyclaw.limiters.tvd.minmod
        solver.max_steps = 5000000
        solver.dq_src = None
        solver.call_before_step_each_stage = before_stage


        
    elif solver_type=='classic':
        solver = pyclaw.ClawSolver1D(rs)
        solver.order = order
        solver.source_split = order
        solver.limiters = pyclaw.limiters.tvd.minmod
        solver.cfl_max = 0.9
        solver.cfl_desired =CFL
        solver.max_steps = 5000000
        if far_field_damping:
            if scalar_far_field:
                solver.step_source = clawpack_source_step_far_field_scalar
            elif implicit_integrator:
                solver.step_source = clawpack_RK23_source_step_far_field
            else:
                solver.step_source = clawpack_source_step_far_field
        elif relaxation_method:
            if matrix_filter:
                solver.step_source = clawpack_source_step_relaxation_matrix
            else:
                solver.step_source = clawpack_source_step_relaxation_scalar
        
    
    #Number of equations and waves
    num_eqn = 3
    num_waves = 2
    
    #Spatial domain
    xlower = 0.0
    xupper = xmax

    solver.kernel_language = kernel_language
    

    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain,num_eqn,2)
    state.problem_data['gamma'] = gamma
    
    #################################################
    #Defining before step functions
    #################################################

    def b4step_relaxation_method(solver,state):
        #Relaxation method
        x = state.grid.x.centers
        ####
        #First we impose constant extrapolation artificially at the right boundary
        indx_const_extrap = max(np.where(x<min(stop_slowing,stop_absorbing))[0])
        state.q[:,indx_const_extrap:] = state.q[:,indx_const_extrap][:,np.newaxis]
        ####
        V, u, eps = state.q[0,:], state.q[1,:], state.q[2,:]
        p = (gamma-1)*(eps-0.5*u**2)/V
        V_star = np.ones(len(V))
        u_star = np.zeros(len(u))
        eps_star = (1./(gamma*(gamma-1)))*np.ones(len(eps))

        #\Gamma(x) used to build the weight function
        #Goes from 1 to 0 in the interval [xL,xR]
        c = state.aux[1].copy()
        if adaptiveRM:
            c = c**solver.dt

        if matrix_filter:
            #Building the matrix M=R\Lambda*|Lambda|^{-1}R^{-1},
            #where R is the matrix of right eigenvectors of the Euler equations
            #and \Lambda is the diagonal matrix of eigenvalues
            
            #Damping coefficients
            d1 = np.ones(len(x))
            d2 = np.ones(len(x))
            d3 = c.copy()

            M = np.zeros((3,3,len(x)))
            spv = np.sqrt(p*V)
            sg = np.sqrt(gamma)
            A = d1-2.*d2+d3
            B = d1-d3
            C = gamma-1.
            J = 2*sg*spv
            
            #First row
            M[0,0,:] = (0.5/gamma)*(2*gamma*d2+A)
            M[0,1,:] = (0.5/p/gamma)*(sg*spv*B+u*C*A)
            M[0,2,:] = (0.5/p/gamma)*(-A*C)
            #Second row
            M[1,0,:] = B*p/J
            M[1,1,:] = (sg*spv*(d1+d3)+u*C*B)/J
            M[1,2,:] = -B*C/J
            #Third row
            M[2,0,:] = (-p*V*A+B*spv*sg*u)/(2*V*gamma)
            M[2,1,:] = (spv*u*A+sg*(C*u**2-p*V)*B)/(J*sg)
            M[2,2,:] = (-spv*A-sg*C*B*u+gamma*spv*(d1+d3))/(J*sg)

            #Building the matrix I-M
            I = np.zeros((3,3,len(x)))
            I[0,0,:] = 1.
            I[1,1,:] = 1.
            I[2,2,:] = 1.

            #Set shape q_star
            q_star = np.zeros((3,len(x)))
            q_star[0,:] = V_star
            q_star[1,:] = u_star
            q_star[2,:] = eps_star

            q_temp = np.einsum('ijk,jk->ik',M,state.q)+np.einsum('ijk,jk->ik',I-M,q_star)
            state.q = np.where(c<1., q_temp, state.q)
            
        else:
            state.q[0,:] = c*V+(1-c)*V_star
            state.q[1,:] = c*u+(1-c)*u_star
            state.q[2,:] = c*eps+(1-c)*eps_star


    #################################################
    def b4step_const_extrap(solver,state):
        x = state.grid.x.centers
        ####
        #First we impose constant extrapolation artificially at the right boundary
        indx_const_extrap = max(np.where(x<=min(start_absorbing,start_slowing))[0])
        state.q[:,indx_const_extrap:] = state.q[:,indx_const_extrap][:,np.newaxis]
        ####
    #################################################

    #################################################
    def b4step_hard_BC(solver,state):
        x = state.grid.x.centers
        V_star = 1.
        u_star = 0.
        eps_star = (1./(gamma*(gamma-1)))
        ####
        #First we impose constant extrapolation artificially at the right boundary
        indx_const_extrap = max(np.where(x<=min(stop_absorbing,stop_slowing))[0])
        #state.q[:,indx_const_extrap:] = state.q[:,indx_const_extrap][:,np.newaxis]
        state.q[0,indx_const_extrap:] = V_star
        state.q[1,indx_const_extrap:] = u_star
        state.q[2,indx_const_extrap:] = eps_star
        ####
    #################################################

    if relaxation_method and solver_type=='sharpclaw':
        solver.before_step = b4step_relaxation_method
    elif integral_source and solver_type=='sharpclaw_custom':
        solver.before_step = b4step_hard_BC
    elif constant_extrap:
        solver.before_step = b4step_const_extrap
    else :
        #solver.before_step = b4step_const_extrap
        solver.before_step = b4step_hard_BC

    ###########################################
    #Defining custom BC. Inflow at left boundary.
    ###########################################
    def piston_bc(state,dim,t,qbc,auxbc,num_ghost):
    #"""Initial pulse generated at left boundary by prescribed motion"""
        if dim.on_lower_boundary:
            qbc[0,:num_ghost] = qbc[0,num_ghost]
            qbc[1,:num_ghost] = qbc[1,num_ghost]
            qbc[2,:num_ghost] = qbc[2,num_ghost]
            t = state.t; gamma = state.problem_data['gamma']
            xi = state.grid.x.centers; 
            deltaxi = xi[1]-xi[0]
            [V1,u1,eps1] = state.q[:,0]
            p1 = (gamma-1.)*(eps1-0.5*u1**2)/V1
            p0 = p1 - M*np.cos(piston_freq*t)*deltaxi
            u0 = -2*M*np.sin(piston_freq*t) - u1
            deltap = (p1-p0)/(p1+p0)
            V0 = V1*(gamma +deltap)/(gamma-deltap)
            eps0 = V0*p0/(gamma-1)+0.5*u0**2
            
            for ibc in range(num_ghost-1):
                qbc[0,num_ghost-ibc-1] = V0
                qbc[1,num_ghost-ibc-1] = u0
                qbc[2,num_ghost-ibc-1] = eps0
    ###########################################
    ###########################################
    #Set problem parameters

    solver.bc_lower[0] = pyclaw.BC.custom
    solver.user_bc_lower = piston_bc
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.num_waves = num_waves
    solver.num_eqn = num_eqn
    x = state.grid.x.centers

    ####################################### 
    #######################################
    #Far-field slowing operator
    if euler_RS == 'euler_slowing':
        if sigma_slowing_type=="affine":
            sigma_slowing = affine_filter(x=x,xL=start_slowing,xR=stop_slowing)
        elif sigma_slowing_type=="Mayer":
            sigma_slowing = Mayer_etal_filter(x=x,xL=start_slowing,xR=stop_slowing)
        elif sigma_slowing_type=="Engsig-Karup2":
            sigma_slowing =Engsig_Karup_etal_filter2(x=x,xL=start_slowing,xR=stop_slowing)
        else: 
            raise ValueError("Slowing function type not recognized")
    else:
        sigma_slowing =np.ones(len(x))

    ####################################### 
    #######################################
    
    if far_field_damping or integral_source:
        #Far-field damping operator (passed to the RS but not used if far_field_damping==False)
        #All the filter functions go from 1 to 0 in the interval [xL,xR]
        if sigma_damping_type=="affine":
            sigma_damping = far_field_damping_rate*(1.-affine_filter(x=x,xL=start_absorbing,xR=stop_absorbing))
        elif sigma_damping_type=="Mayer":
            sigma_damping = far_field_damping_rate*(1.-Mayer_etal_filter(x=x,xL=start_absorbing,xR=stop_absorbing,
                                                                         b=b_Mayer_filter))
        elif sigma_damping_type=="Engsig-Karup2":
            sigma_damping = far_field_damping_rate*(1.-Engsig_Karup_etal_filter2(x=x,xL=start_absorbing,xR=stop_absorbing))
        else:
            raise ValueError("Damping function type not recognized")
    elif relaxation_method:
        #Spatial function meant to be used in the relaxation method
        #Used to construct the weight function (in the most basic case, sigma_damping is the weight function itself)
        if sigma_damping_type=="affine":
            sigma_damping = affine_filter(x=x,xL=start_absorbing,xR=stop_absorbing)
        elif sigma_damping_type=="Mayer":
            sigma_damping = Mayer_etal_filter(x=x,xL=start_absorbing,xR=stop_absorbing,
                                                                         b=b_Mayer_filter)
        elif sigma_damping_type=="Engsig-Karup2":
            sigma_damping = Engsig_Karup_etal_filter2(x=x,xL=start_absorbing,xR=stop_absorbing)
        else:
            raise ValueError("Damping function type not recognized")
    else:
        sigma_damping = np.zeros(len(x))

    ####################################### 
    #######################################
    #Set initial conditions
    init(state,sigma_slowing,sigma_damping)

    #Set up controller and controller parameters
    claw = pyclaw.Controller()
    claw.tfinal = tfinal
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = nout
    claw.outdir = outdir
    claw.setplot = setplot
    claw.keep_copy = True

    return claw

####################################### 
#######################################
#Initial conditions
def init(state, sigma_slowing, sigma_damping):
    gamma = state.problem_data['gamma']
    
    rho = 1.
    V = 1./rho
    p = 1./gamma
    u = 0.
    eps = V*p/(gamma-1)+0.5*u**2

    state.q[0 ,:] = V
    state.q[1,:] = u
    state.q[2,:] = eps

    #state.aux[0,:] = x
    state.aux[0,:] = sigma_slowing
    state.aux[1,:] = sigma_damping
####################################### 
#######################################

####################################### 
#######################################
#Filtering functions
#Used to construct the functions sigma_damping and sigma_slowing
def affine_filter(x,xL,xR):
        #Defining relaxation zone
        #Generates an affine function equal to 1 for x<=a and equal to 0 for x>b
        Lrelax = xR-xL
        di = x-xR #distance of points from relaxation zone's origin 
        mi = np.abs(di)/Lrelax #slope of filter in relaxation zone
        return (mi*(mi<=1)+(mi>1))*(di<=0)

def Mayer_etal_filter(x,xL,xR,b=0.5):
    beta_fun = lambda x: np.abs(x-xL)/np.abs(xR-xL)
    sigma_fun = lambda x: 1-b*beta_fun(x)**3-(1-b)*beta_fun(x)**6
    return np.where(x<xL,1.,np.where(x>xR,0.,sigma_fun(x)))

def Engsig_Karup_etal_filter2(x,xL,xR):  
    #Decays fast from 1 to 0, starts decreasing from the beginning  
    phimap = lambda x: np.where(x<xL,0.,np.where(x>xR,1.,(x-xL)/(xR-xL)))
    sigma_fun = lambda x:-2*(1-x)**3+3*(1-x)**2
    return sigma_fun(phimap(x))
#######################################
#######################################


#Plotting functions
#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data
    xmin, xmax = 0., 400.
    plotfigure = plotdata.new_plotfigure(name='', figno=0)

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(211)'
    plotaxes.xlimits = [xmin,xmax]
    plotaxes.title = 'V'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 0#density
    plotitem.kwargs = {'linewidth':1}
    
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = [xmin,xmax]
    plotaxes.ylimits = [-1.5,1.5]
    plotaxes.title = 'u'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 1#energy
    plotitem.kwargs = {'linewidth':1}
    
    return plotdata
#######################################
#######################################

#Main function
if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
