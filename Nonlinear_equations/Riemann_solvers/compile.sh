#!/bin/bash

#python3 -m numpy.f2py -c rp1_euler_HLL.f90 -m euler_HLL_1D
#python3 -m numpy.f2py -c rp1_euler_HLL_slowing.f90 -m euler_HLL_slowing_1D
#python3 -m numpy.f2py -c rp1_euler_HLL_slowing_damping.f90 -m euler_HLL_slowing_damping_1D
#python3 -m numpy.f2py -c rp1_euler_burgers_HLL.f90 -m euler_burgers_HLL_1D

#Change to the directory with the Fortran Riemann solvers
cd ./src

#Iterate over all files in the directory
for file in *.f90
do
    #Get the name of the file without the extension
    name=${file%.f90}
    #Strip the first appearance of 'rp1_'
    name=${name#rp1_}
    #ADD 1D
    name=${name}_1D

    #Compile the file
    python3 -m numpy.f2py -c $file -m $name
done

#Clean the ../modules directory
rm ../modules/*.so
#Move each compiled file to the ../modules directory
mv *.so ../modules

#Change back to the directory with the Python scripts
cd ..