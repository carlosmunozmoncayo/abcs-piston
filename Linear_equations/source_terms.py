from __future__ import absolute_import
from __future__ import print_function
from clawpack import riemann
import numpy as np
import sys

#######################################
#######################################
#Far field damping
def sharpclaw_source_step_damping(solver,state,dt):
    xc = state.grid.x.centers

    #Get parameters
    gamma = state.problem_data['gamma']
    sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]

    #Damping coefficients
    d1 = np.zeros_like(xc)#np.ones(len(x))
    d2 = np.zeros_like(xc)
    d3 = gamma*sigma_damping*sigma_slowing

    #Building Matrix M=RDR^{-1}
    #to solve the system W_t = -M W
    M = np.zeros((3,3,len(xc)))
    #First row
    M[0,0,:] = (1/gamma)*((d1+d3)/2+(gamma-1)*d2)
    M[0,1,:] = (0.5/gamma)*(d1-d3)
    M[0,2,:] = ((gamma-1)/gamma**2)*(0.5*(-d1-d3)+d2)
    #Second row
    M[1,0,:] = 0.5*(d1-d3)
    M[1,1,:] = 0.5*(d1+d3)
    M[1,2,:] = 0.5*((gamma-1)/gamma)*(d3-d1)
    #Third row
    M[2,0,:] = -0.5*(d1+d3)+d2
    M[2,1,:] = 0.5*(d3-d1)
    M[2,2,:] = (1./gamma)*(0.5*(gamma-1)*(d1+d3)+d2)

    dq = -dt*np.einsum('ijk,jk->ik',M,state.q)
    return dq
#######################################
#######################################

#######################################
#######################################
#Far field damping scalar
def sharpclaw_source_step_damping_scalar(solver,state,dt):
    xc = state.grid.x.centers

    #Get parameters
    gamma = state.problem_data['gamma']
    sigma_damping = state.aux[1]
    sigma_slowing = state.aux[0]

    #Building Matrix A=RDR^{-1}
    #to solve the system W_t = -d A W
    A = np.zeros((3,3,len(xc)))
    #First row
    A[0,0,:] = 0.0
    A[0,1,:] = -1.0
    A[0,2,:] = 0.0
    #Second row
    A[1,0,:] = -gamma
    A[1,1,:] = 0.0
    A[1,2,:] = gamma-1.0
    #Third row
    A[2,0,:] = 0.0
    A[2,1,:] = gamma
    A[2,2,:] = 0.0

    d = sigma_damping
    s = sigma_slowing
    dq = -dt*d*s*np.einsum('ijk,jk->ik',A,state.q)
    return dq
#######################################
#######################################


#######################################
#######################################
def clawpack_source_step_far_field(solver,state,dt):
    q = state.q.copy()
    xc = state.grid.x.centers
    #Get parameters
    gamma = state.problem_data['gamma']
    #sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]
    d1 = np.ones_like(xc)#np.ones(len(x))
    d2 = np.ones_like(xc)
    d3 = np.exp(-dt*gamma*sigma_damping)

    #Building Matrix exp(-dt*M)= R exp(-dt*D) R^{-1}
    #to solve the system W_t = -M W
    M = np.zeros((3,3,len(xc)))
    #First row
    M[0,0,:] = (1/gamma)*((d1+d3)/2+(gamma-1)*d2)
    M[0,1,:] = (0.5/gamma)*(d1-d3)
    M[0,2,:] = ((gamma-1)/gamma**2)*(0.5*(-d1-d3)+d2)
    #Second row
    M[1,0,:] = 0.5*(d1-d3)
    M[1,1,:] = 0.5*(d1+d3)
    M[1,2,:] = 0.5*((gamma-1)/gamma)*(d3-d1)
    #Third row
    M[2,0,:] = -0.5*(d1+d3)+d2
    M[2,1,:] = 0.5*(d3-d1)
    M[2,2,:] = (1./gamma)*(0.5*(gamma-1)*(d1+d3)+d2)
    
    state.q = np.einsum('ijk,jk->ik',M,q)

#######################################
#######################################

#######################################
#######################################
def clawpack_source_step_far_field_scalar(solver,state,dt):
    q = state.q.copy()
    xc = state.grid.x.centers
    #Get parameters
    gamma = state.problem_data['gamma']
    #sigma_slowing = state.aux[0]
    sigma_damping = state.aux[1]

    #Heun's method
    state.q = np.exp(-dt*sigma_damping)*q

#######################################
#######################################

#######################################
#######################################
def clawpack_source_step_relaxation_scalar(solver,state,dt,adaptiveRM=False,Strang=False):
    gamma = state.problem_data['gamma']
    c = state.aux[1].copy()
    if adaptiveRM:
        c = c**dt
    if Strang:
        c = c**0.5
    
    for i in range(3):
        state.q[i,:] = c*state.q[i,:]


#######################################
#######################################

#######################################
#######################################

def clawpack_source_step_relaxation_matrix(solver,state,dt,adaptiveRM=False,Strang=False):
    gamma = state.problem_data['gamma']
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
    #First row
    M[0,0,:] = (1/gamma)*((d1+d3)/2+(gamma-1)*d2)
    M[0,1,:] = (0.5/gamma)*(d1-d3)
    M[0,2,:] = ((gamma-1)/gamma**2)*(0.5*(-d1-d3)+d2)
    #Second row
    M[1,0,:] = 0.5*(d1-d3)
    M[1,1,:] = 0.5*(d1+d3)
    M[1,2,:] = 0.5*((gamma-1)/gamma)*(d3-d1)
    #Third row
    M[2,0,:] = -0.5*(d1+d3)+d2
    M[2,1,:] = 0.5*(d3-d1)
    M[2,2,:] = (1./gamma)*(0.5*(gamma-1)*(d1+d3)+d2)

    state.q = np.einsum('ijk,jk->ik',M,state.q)

#######################################
#######################################