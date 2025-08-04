import matplotlib.pyplot as plt
import numpy as np
from clawpack import pyclaw
from piston_Lagrangian import setup


piston_freq = 1
p_domain = 30
p = 20
#Reference solution
xmax1 = p_domain*2*np.pi 
N = 250*12 #Cells per period at final time
M = 0.4
mx1 = p_domain*N #Number of grid points
tfinal = p*2*np.pi
print("Reference solution will have {} cells".format(mx1))

#Run reference solution
piston_problem1 = setup(outdir="./data_convergence",tfinal=tfinal, xmax=xmax1, mx=mx1, M=M, CFL=0.8,limiting=1,solver_type='sharpclaw',
order=2, time_integrator='Heun', nout=100, piston_freq=piston_freq)
# piston_problem1.verbosity = 0
piston_problem1.run()
piston_problem1.solver.__del__()