from __future__ import absolute_import
from clawpack import riemann
from clawpack import pyclaw
import numpy as np
import sys

#Importing clawpack source terms
from source_terms import clawpack_source_step_relaxation_scalar, clawpack_source_step_relaxation_matrix
from source_terms import clawpack_source_step_far_field

from source_terms import clawpack_source_step_far_field_scalar, clawpack_RK23_source_step_far_field

#Import clawpack integral source term
from source_terms import clawpack_source_step_nonlinear_far_field

#Import sharpclaw source term
from source_terms import sharpclaw_source_step_damping, sharpclaw_source_step_damping_scalar



class customSharpClawSolver1D(pyclaw.SharpClawSolver1D):
    def __init__(self,riemann_solver=None,claw_package='clawpack.pyclaw',
                far_field_damping = False, 
                relaxation_method = False, 
                scalar_far_field =False,
                implicit_integrator=False,
                matrix_filter=False,
                integral_source=False,
                adaptiveRM=False,
                Strang=False):
        self.far_field_damping = far_field_damping
        self.relaxation_method = relaxation_method
        self.matrix_filter = matrix_filter
        self.scalar_far_field = scalar_far_field
        self.implicit_integrator = implicit_integrator
        self.integral_source = integral_source
        self.adaptiveRM = adaptiveRM
        self.Strang = Strang
        super().__init__(riemann_solver=riemann_solver,claw_package=claw_package)

    def custom_source_step(self,state,dt):
        """To integrate the source term manually, we can recycle the functions that
            we have already written for the classic Clawpack source terms.
            This solver class will use Strang operator splitting to integrate the source term.
        """
        if self.far_field_damping:
            if self.scalar_far_field:
                clawpack_source_step_far_field_scalar(self,state,dt)
            elif self.implicit_integrator:
                clawpack_RK23_source_step_far_field(self,state,dt)
            else:
                clawpack_source_step_far_field(self,state,dt)

        elif self.relaxation_method:
            if self.matrix_filter:
                clawpack_source_step_relaxation_matrix(self,state,dt,adaptiveRM=self.adaptiveRM,Strang=self.Strang)
            else:
                clawpack_source_step_relaxation_scalar(self,state,dt,adaptiveRM=self.adaptiveRM,Strang=self.Strang)
        
        elif self.integral_source:
            clawpack_source_step_nonlinear_far_field(self,state,dt)
        else:
            pass #Do nothing if there is no source
    def step(self,solution,take_one_step,tstart,tend):
        """Evolve q over one time step.

        Take one step with a Runge-Kutta or multistep method as specified by
        `solver.time_integrator`.

        If self.time_integrator is 'RK', we will handle the source term with operator splitting.
        For this, it is assumed that solver.dq_src=None. We will force the latter just in case.
        """
        state = solution.states[0]

        ###########
        ###########
        self.dq_src = None #Force dq_src to be None
        ###########
        ###########

        step_index = self.status['numsteps'] + 1
        if self.accept_step == True:
            self.cfl.set_global_max(0.)
            self.dq_dt = self.dq(state) / self.dt

        if 'LMM' in self.time_integrator:
            step_index = self.update_saved_values(state,step_index)

        self.get_dt(solution.t,tstart,tend,take_one_step)

        # Recompute cfl number based on current step-size
        cfl = self.cfl.get_cached_max()
        self.cfl.set_global_max(self.dt / self.dt_old * cfl)
        self.dt_old = self.dt

        ### Runge-Kutta methods ###
        if self.time_integrator == 'Euler':
            state.q += self.dt*self.dq_dt

        elif self.time_integrator == 'SSP22':
            self.ssp22(state)

        elif self.time_integrator == 'SSP33':
            self._registers[0].q = state.q + self.dt*self.dq_dt
            self._registers[0].t = state.t + self.dt

            if self.call_before_step_each_stage:
                self.before_step(self,self._registers[0])
            self._registers[0].q = 0.75*state.q + 0.25*(self._registers[0].q + self.dq(self._registers[0]))
            self._registers[0].t = state.t + 0.5*self.dt

            if self.call_before_step_each_stage:
                self.before_step(self,self._registers[0])
            state.q = 1./3.*state.q + 2./3.*(self._registers[0].q + self.dq(self._registers[0]))

        elif self.time_integrator == 'SSP104':
            self.ssp104(state)

        elif self.time_integrator == 'RK':
            # General explicit RK with specified coefficients for the hyperbolic part
            # and operator splitting for the source term.

            if self.Strang:
                ###########
                self.custom_source_step(state=state,dt=self.dt/2.) #First half of the Strang splitting
                ###########
            else:
                ###########
                self.custom_source_step(state=state,dt=self.dt) #Lie-Trotter
                ###########
            
            # This is pulled out of the loop in order to use dq_dt
            self._registers[0].q = self.dt*self.dq_dt
            self._registers[0].t = state.t

            num_stages = len(self.b)
            for i in range(1,num_stages):
                self._registers[i].q[:] = state.q
                self._registers[i].t = state.t + self.dt*self.c[i]
                if self.call_before_step_each_stage:
                    self.before_step(self,self._registers[i])
                    self.custom_source_step(state=state,dt=self.dt) #Using the RM at each stage
                for j in range(i):
                    self._registers[i].q += self.a[i,j]*self._registers[j].q

                # self._registers[i].q eventually stores dt*f(y_i) after stage solution y_i is computed
                self._registers[i].q = self.dq(self._registers[i])

            for j in range(num_stages):
                state.q += self.b[j]*self._registers[j].q

            if self.Strang:
                ###########
                self.custom_source_step(state=state,dt=self.dt/2.) #Second half of the Strang splitting
                ###########

        ### Linear multistep methods ###
        elif self.time_integrator in ['SSPLMMk2', 'SSPLMMk3']:
            num_steps = self.lmm_steps
            if step_index < num_steps:
                # Use SSP22 Runge-Kutta method for starting values
                self.ssp22(state)
            else:
                if self.time_integrator == 'SSPLMMk2':
                    omega_k_minus_1 = sum(self.prev_dt_values[1:])/self.dt
                    r = (omega_k_minus_1-1.)/omega_k_minus_1 # SSP coefficient

                    delta = 1./omega_k_minus_1**2
                    beta = (omega_k_minus_1+1.)/omega_k_minus_1
                    state.q = beta*(r*state.q + self.dt*self.dq_dt) + delta*self._registers[-num_steps].q
                else:
                    omega_k_minus_1 = sum(self.prev_dt_values[1:])/self.dt
                    omega_k = omega_k_minus_1 + 1.
                    r = (omega_k_minus_1-2.)/omega_k_minus_1 # SSP coefficient

                    delta0 = (4*omega_k - omega_k_minus_1**2)/omega_k_minus_1**3
                    beta0 = omega_k/omega_k_minus_1**2
                    beta_k_minus_1 = omega_k**2/omega_k_minus_1**2

                    state.q = beta_k_minus_1*(r*state.q + self.dt*self.prev_dq_dt_values[-1]) + \
                            (r*beta0 + delta0)*self._registers[-num_steps].q + \
                            beta0*self.dt*self.prev_dq_dt_values[-num_steps]

        elif self.time_integrator == 'LMM':
            if step_index < len(self._registers):
                self.ssp104(state) # Use SSP104 for starting values
            else:
                # Update solution: alpha[-1] and beta[-1] correspond to solution at the previous step
                state.q = self.alpha[-1]*self._registers[-1].q + self.beta[-1]*self.dt*self.prev_dq_dt_values[-1]
                for i in range(self.lmm_steps-1):
                    state.q += self.alpha[i]*self._registers[i].q + self.beta[i]*self.dt*self.prev_dq_dt_values[i]

        else:
            raise Exception('Unrecognized time integrator')
            return False
    pass