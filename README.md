A Unified Framework for Sponge-Layer Relaxation Methods and Damping Operators for Conservation Laws: Application to the Piston Problem of Gas Dynamics
===========================================================================
Reproducibility repository for the paper [A Unified Framework for Sponge-Layer Relaxation Methods and Damping Operators for Conservation Laws: Application to the Piston Problem of Gas Dynamics](https://link.springer.com/article/10.1007/s10665-025-10504-0)
![image](https://github.com/carlosmunozmoncayo/abcs-piston/assets/29715468/afc55443-2cd1-4df8-8cbb-f07984e4abb2)
![image](https://github.com/carlosmunozmoncayo/abcs-piston/assets/29715468/df93afef-9752-4653-a76f-4275e2fc1c4c)

### Dependencies
  - python (v3.8.9)
  - numpy (v1.23.3)
  - matplotlib (v3.6.3)
  - f2py (v1.22.3)
  - A Fortran compiler (GNU Fortran compiler v11.2.0)
  - Clawpack (v5.9.2)

The Jupyter Notebooks contain descriptions of their uses. Before running them, the Riemann solvers and the code for the NDO ABC must be compiled with f2py. For this, it suffices to run the bash scripts **compile.sh** in the corresponding directories.
