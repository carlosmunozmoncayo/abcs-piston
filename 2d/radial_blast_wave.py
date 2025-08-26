#!/usr/bin/env python
# encoding: utf-8
r"""
Euler 2D Quadrants example
==========================

Simple example solving the Euler equations of compressible fluid dynamics:

.. math::
    \rho_t + (\rho u)_x + (\rho v)_y & = 0 \\
    (\rho u)_t + (\rho u^2 + p)_x + (\rho uv)_y & = 0 \\
    (\rho v)_t + (\rho uv)_x + (\rho v^2 + p)_y & = 0 \\
    E_t + (u (E + p) )_x + (v (E + p))_y & = 0.

Here :math:`\rho` is the density, (u,v) is the velocity, and E is the total energy.
The initial condition is one of the 2D Riemann problems from the paper of
Liska and Wendroff.

"""
from clawpack import riemann
from clawpack.riemann.euler_4wave_2D_constants import density, x_momentum, y_momentum, \
        energy, num_eqn
from clawpack.visclaw import colormaps
import numpy as np
import bcs_aux

def setplot(plotdata):
    r"""Plotting settings

    Should plot two figures both of density.

    """


    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for density - pcolor
    plotfigure = plotdata.new_plotfigure(name='Density', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.scaled = True
    plotaxes.title = 'Density'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = density
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.pcolor_cmin = 0.3
    plotitem.pcolor_cmax = 1.3
    plotitem.add_colorbar = True

    # Figure for density - Schlieren
    plotfigure = plotdata.new_plotfigure(name='Schlieren', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = 'Density'
    plotaxes.scaled = True      # so aspect ratio is 1

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_schlieren')
    plotitem.schlieren_cmin = 0.0
    plotitem.schlieren_cmax = 1.0
    plotitem.plot_var = density
    plotitem.add_colorbar = False
    
    return plotdata


def setup(use_petsc=False,riemann_solver='roe',tfinal=2.0):
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if riemann_solver.lower() == 'roe':
        solver = pyclaw.ClawSolver2D(riemann.euler_4wave_2D)
        solver.transverse_waves = 2
    elif riemann_solver.lower() == 'hlle':
        solver = pyclaw.ClawSolver2D(riemann.euler_hlle_2D)
        solver.transverse_waves = 0
        solver.cfl_desired = 0.4
        solver.cfl_max = 0.5
    solver.all_bcs = pyclaw.BC.wall

    domain = pyclaw.Domain([-2.,-2.],[2.,2.],[300,300])
    # solution = pyclaw.Solution(num_eqn,domain)
    state = pyclaw.State(domain,num_eqn)
    gamma = 1.4
    state.problem_data['gamma']  = gamma

    # Set initial data
    xx, yy = domain.grid.p_centers
    radius = ((xx)**2 + (yy)**2)**0.5
    state.q[density,...] = 1.
    u = 0.0
    v = 0.0
    p = np.where(radius<0.2,5*state.q[density,...],state.q[density,...])
    state.q[x_momentum,...] = state.q[density, ...] * u
    state.q[y_momentum,...] = state.q[density, ...] * v
    state.q[energy,...] = 0.5 * state.q[density,...]*(u**2 + v**2) + p / (gamma - 1.0)

    
    #####################################################################
    #Set RM-based ABCs 
    #####################################################################
    # xvec = xx[:,0]; yvec = yy[0,:]
    # (idx_lims, sponge_slices,
    #   theta_slices, rel_funcs) = bcs_aux.setup_RM(xvec,yvec,[xvec[0],-1.8,1.8,xvec[-1]],
    #                                               [yvec[0],-1.8,1.8,yvec[-1]])
    # state.idx_lims = idx_lims
    # state.sponge_slices = sponge_slices
    # state.theta_slices = theta_slices
    # state.rel_funcs = rel_funcs
    # state.q_target = state.q[:,-1,-1].copy()  # Target state is state at the boundary at initial time
    # def b4step_RM(solver,state):
    #     r"""
    #     Function to be called before each time step to apply RM ABCs.
    #     """
    #     state.q = bcs_aux.apply_RM(state.q,state.rel_funcs,state.q_target,state.idx_lims)
    # solver.before_step = b4step_RM
    #####################################################################
    #####################################################################
    #Set RM-M-based ABCs 
    xvec = xx[:,0]; yvec = yy[0,:]
    (idx_lims, sponge_slices,
      theta_slices, rel_funcs) = bcs_aux.setup_RM(xvec,yvec,[xvec[0],-1.8,1.8,xvec[-1]],
                                                  [yvec[0],-1.8,1.8,yvec[-1]])
    state.idx_lims = idx_lims
    state.sponge_slices = sponge_slices
    state.theta_slices = theta_slices
    state.rel_funcs = rel_funcs
    state.q_target = state.q[:,-1,-1].copy()  # Target state is state at the boundary at initial time
    def b4step_RMM(solver,state):
        r"""
        Function to be called before each time step to apply RM ABCs.
        """
        state.q = bcs_aux.apply_RMM(state.q,state.rel_funcs,state.theta_slices,state.q_target,gamma,state.idx_lims)
    solver.before_step = b4step_RMM


    #####################################################################


    claw = pyclaw.Controller()
    claw.tfinal = tfinal
    claw.num_output_times = 10
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver

    claw.output_format = 'ascii'    
    claw.outdir = "./_output"
    claw.keep_copy = True
    claw.setplot = setplot

    return claw

if __name__ == "__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup, setplot)
