from __future__ import absolute_import
from __future__ import print_function
from clawpack import riemann
import numpy as np
from scipy import optimize
from scipy.integrate import solve_ivp
import sys
sys.path.append('./Fortran_source_terms/modules')
import nonlinear_ABCs_module


gamma = 1.4 # Ratio of specific heats


#######################################
#######################################
#Far field damping
def sharpclaw_source_step_damping(solver,state,dt):
    #Time integration of the system W_t+ D W = 0
    #where D is given by eq. (29) of Karni's (1996)
    q = state.q
    xc = state.grid.x.centers

    #Get parameters
    gamma = state.problem_data['gamma']
    sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]

    #Get variables
    V = q[0,:]
    sv = np.sqrt(V)
    u = q[1,:]
    eps = q[2,:]

    #Defining some variables
    p = (gamma-1.)*(eps-0.5*u**2)/V
    sp = np.sqrt(p)
    spv = sp*sv

    #Right eigenvalue
    lamb3 = np.sqrt(gamma)*sp/sv

    #Constant steady state
    V0 = 1.
    u0 = 0.
    p0 = 1/gamma
    eps0 = p0*V0/(gamma-1.)+0.5*u0**2
    #define q0 as a 3xN array with q0[:,i] = (V0,u0,eps0)
    q0 = np.zeros((3,len(V)))
    q0[0,:] = V0
    q0[1,:] = u0
    q0[2,:] = eps0

    q_goal = q-q0

    #############
    ###Computing damping matrix M=R\Lambda R^{-1}
    #############

    #Damping coefficients
    d1 = np.zeros_like(V)
    d2 = np.zeros_like(V)
    d3= sigma_damping*lamb3*sigma_slowing 

    M = np.zeros((3,3,len(xc)))
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
    #############
    
    #Damping source term
    dq = np.einsum('abm,bm->am',-M,q_goal)
    dq = dt*np.where(sigma_damping>1.e-10,dq,0.)
    return dq
#######################################
#######################################

#######################################
#######################################
#Far field damping scalar
def sharpclaw_source_step_damping_scalar(solver,state,dt):
    #Time integration of the system W_t+ d f'(W) W = 0
    #where D is given by eq. (29) of Karni's (1996)
    q = state.q
    xc = state.grid.x.centers

    #Get parameters
    gamma = state.problem_data['gamma']

    sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]
    #Get variables
    V = q[0,:]
    sv = np.sqrt(V)
    u = q[1,:]
    eps = q[2,:]

    #Defining some variables
    p = (gamma-1.)*(eps-0.5*u**2)/V
    sp = np.sqrt(p)
    spv = sp*sv

    #Constant steady state
    V0 = 1.
    u0 = 0.
    p0 = 1/gamma
    eps0 = p0*V0/(gamma-1.)+0.5*u0**2
    #define q0 as a 3xN array with q0[:,i] = (V0,u0,eps0)
    q0 = np.zeros_like(q)
    q0[0,:] = V0
    q0[1,:] = u0
    q0[2,:] = eps0

    M = np.zeros((3,3,len(xc)))
    #First row
    M[0,0,:] = 0.0
    M[0,1,:] = -1.0
    M[0,2,:] = 0.0
    #Second row
    M[1,0,:] = -p/V
    M[1,1,:] = u*(1.0-gamma)/V
    M[1,2,:] = (gamma-1.0)/V
    #Third row
    M[2,0,:] = -p*u/V
    M[2,1,:] = p+u**2*(1.0-gamma)/V
    M[2,2,:] = u*(-1.0+gamma)/V
    #############

    q_goal = q-q0
    #Damping source term
    dq = np.einsum('abm,bm->am',M,q_goal)
    dq = -dt*sigma_damping*sigma_slowing*np.where(sigma_damping>1.e-10,dq,0.)
    return dq
#######################################
#######################################

#######################################
#######################################
def clawpack_source_step_far_field(solver,state,dt):
    "Integrate the source term over one step."
    q = state.q.copy()
    xc = state.grid.x.centers
    #Get parameters
    gamma = state.problem_data['gamma']
    #sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]

    def clawpack_far_field_RHS(q_local=q, gamma=gamma, sigma_damping=sigma_damping):
        #Get variables
        V = q_local[0,:]
        sv = np.sqrt(V)
        u = q_local[1,:]
        eps = q_local[2,:]

        #Defining some variables
        p = (gamma-1.)*(eps-0.5*u**2)/V
        sp = np.sqrt(p)
        spv = sp*sv

        #Right eigenvalue
        lamb3 = np.sqrt(gamma)*sp/sv

        #Constant steady state
        V0 = 1.
        u0 = 0.
        p0 = 1/gamma
        eps0 = p0*V0/(gamma-1.)+0.5*u0**2
        #define q0 as a 3xN array with q0[:,i] = (V0,u0,eps0)
        q0 = np.zeros((3,len(V)))
        q0[0,:] = V0
        q0[1,:] = u0
        q0[2,:] = eps0

        q_local_goal = q_local-q0

        #############
        ###Computing damping matrix M=R\Lambda R^{-1}
        #############

        #Damping coefficients
        d1 = np.zeros_like(V)
        d2 = np.zeros_like(V)
        d3= sigma_damping*lamb3 

        M = np.zeros((3,3,len(xc)))
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
        #############
        return np.einsum('abm,bm->am',-M,q_local_goal)

    #Heun's method
    q_tilde = q + dt*clawpack_far_field_RHS(q_local=q)
    q = q + 0.5*dt*(clawpack_far_field_RHS(q_local=q)+clawpack_far_field_RHS(q_tilde))
    state.q = q.copy()
#######################################
#######################################

#######################################
#######################################
def clawpack_source_step_far_field_scalar(solver,state,dt):
    "Integrate the source term over one step."
    q = state.q.copy()
    xc = state.grid.x.centers
    #Get parameters
    gamma = state.problem_data['gamma']
    #sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]

    #Constant steady state
    V0 = 1.
    u0 = 0.
    p0 = 1/gamma
    eps0 = p0*V0/(gamma-1.)+0.5*u0**2

    q0 = np.zeros_like(q)
    q0[0,:] = V0
    q0[1,:] = u0
    q0[2,:] = eps0

    #We integrate exactly q'=M(q-q0) where M is diagonal with sigma_damping on the diagonal
    state.q = np.exp(-dt*sigma_damping)*(q-q0)+q0
#######################################
#######################################

#######################################
#######################################
def clawpack_source_step_relaxation_scalar(solver,state,dt,adaptiveRM=False,Strang=False):
    gamma = state.problem_data['gamma']
    V, u, eps = state.q[0,:], state.q[1,:], state.q[2,:]
    p = (gamma-1)*(eps-0.5*u**2)/V
    V_star = np.ones(len(V))
    u_star = np.zeros(len(u))
    eps_star = (1./(gamma*(gamma-1)))*np.ones(len(eps))

    #\Gamma(x) used to build the weight function
    #Goes from 1 to 0 in the interval [xL,xR]
    c = state.aux[1].copy()
    if adaptiveRM:
        c = c**dt
    if Strang:
        c = c**0.5

    state.q[0,:] = c*V+(1-c)*V_star
    state.q[1,:] = c*u+(1-c)*u_star
    state.q[2,:] = c*eps+(1-c)*eps_star
#######################################
#######################################

#######################################
#######################################
def clawpack_source_step_relaxation_matrix(solver,state,dt,adaptiveRM=False,Strang=False):
    gamma = state.problem_data['gamma']
    V, u, eps = state.q[0,:], state.q[1,:], state.q[2,:]
    p = (gamma-1)*(eps-0.5*u**2)/V
    V_star = np.ones(len(V))
    u_star = np.zeros(len(u))
    eps_star = (1./(gamma*(gamma-1)))*np.ones(len(eps))

    #\Gamma(x) used to build the weight function
    #Goes from 1 to 0 in the interval [xL,xR]
    c = state.aux[1]
    #Building the matrix M=R\Lambda*|Lambda|^{-1}R^{-1},
    #where R is the matrix of right eigenvectors of the Euler equations
    #and \Lambda is the diagonal matrix of eigenvalues
    
    #Damping coefficients
    d1 = np.ones(len(c))
    d2 = np.ones(len(c))
    d3 = c.copy()

    if adaptiveRM:
        d3 = d3**dt
    if Strang:
        d3 = d3**0.5
            

    M = np.zeros((3,3,len(c)))
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
    I = np.zeros((3,3,len(c)))
    I[0,0,:] = 1.
    I[1,1,:] = 1.
    I[2,2,:] = 1.

    #Set shape q_star
    q_star = np.zeros((3,len(c)))
    q_star[0,:] = V_star
    q_star[1,:] = u_star
    q_star[2,:] = eps_star

    q_temp = np.einsum('ijk,jk->ik',M,state.q)+np.einsum('ijk,jk->ik',I-M,q_star)
    state.q = np.where(c<1., q_temp, state.q)
#######################################
#######################################

#######################################
#######################################
def clawpack_RK23_source_step_far_field(solver,state,dt):
    "Integrate the source term over one step."
    q = state.q.copy()
    xc = state.grid.x.centers
    #Get parameters
    gamma = state.problem_data['gamma']
    #sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]

    #First we make q flat
    q = q.flatten()
    #We define the RHS of the ODE
    def clawpack_far_field_RHS(t,q_local):
        #We reshape q (it will be flattened as argument)
        q_local = q_local.reshape((3,len(xc)))
        #Get variables
        V = q_local[0,:]
        sv = np.sqrt(V)
        u = q_local[1,:]
        eps = q_local[2,:]

        #Defining some variables
        p = (gamma-1.)*(eps-0.5*u**2)/V
        sp = np.sqrt(p)
        spv = sp*sv

        #Right eigenvalue
        lamb3 = np.sqrt(gamma)*sp/sv

        #Constant steady state
        V0 = 1.
        u0 = 0.
        p0 = 1/gamma
        eps0 = p0*V0/(gamma-1.)+0.5*u0**2
        #define q0 as a 3xN array with q0[:,i] = (V0,u0,eps0)
        q0 = np.zeros((3,len(V)))
        q0[0,:] = V0
        q0[1,:] = u0
        q0[2,:] = eps0

        q_local_goal = q_local-q0

        #############
        ###Computing damping matrix M=R\Lambda R^{-1}
        #############

        #Damping coefficients
        d1 = np.zeros_like(V)
        d2 = np.zeros_like(V)
        d3= sigma_damping*lamb3 

        M = np.zeros((3,3,len(xc)))
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
        #############
        return (np.einsum('abm,bm->am',-M,q_local_goal)).flatten()

    sol = solve_ivp(fun=clawpack_far_field_RHS,t_span=[0.,dt],y0=q,method='RK23',rtol=1e-6,atol=1e-6)
    state.q = sol.y[:,-1].reshape((3,len(xc)))
#######################################
#######################################

    
#######################################
#######################################
def clawpack_source_step_nonlinear_far_field(solver,state,dt):
    #Using integral far field source term
    "Integrate the source term over one step."
    q = state.q.copy()
    xc = state.grid.x.centers
    #Get parameters
    gamma = state.problem_data['gamma']
    #sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]

    state.q = nonlinear_ABCs_module.heun_step(q=q, n=len(q[0]), gamma=gamma, dt=dt, sigma_damping_vec=sigma_damping)
#######################################
#######################################


#######################################
#######################################
def sharpclaw_source_step_nonlinear_far_field(solver,state,dt):
    #Using integral far field source term
    "Integrate the source term over one step."
    q = state.q.copy()
    xc = state.grid.x.centers
    #Get parameters
    gamma = state.problem_data['gamma']
    #sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]

    dq = nonlinear_ABCs_module.rhs_2nd_order(q=q, n=len(q[0]), gamma=gamma, sigma_damping_vec=sigma_damping, x=xc)
    return dt*dq
#######################################
#######################################
