#!/bin/bash

#Change to the directory with the Fortran source
cd ./src

#Iterate over all files in the directory
for file in *.f90
do
    #Get the name of the file without the extension
    name=${file%.f90}
    #ADD module
    name=${name}_module

    #Compile the file
    python3 -m numpy.f2py -c $file -m $name
done

#Clean the ../modules directory
rm ../modules/*.so
#Move each compiled file to the ../modules directory
mv *.so ../modules

#Change back to the directory with the Python scripts
cd ..