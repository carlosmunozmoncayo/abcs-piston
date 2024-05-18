import numpy as np

def error (u1,u2):
    #u1 will be a reference solution
    #u2 will be a 'numerical' solution
    return np.sum(np.abs(u1-u2))/np.sum(np.abs(u2))

def max_error(piston_problem1,piston_problem2, indx=[]):
    #Check for Nans in piston_problem2.frames[-1].q[1]
    if np.isnan(piston_problem2.frames[-1].q[0]).any():
        return np.nan
    max_error_absorbing = 0
    for i in range(1,len(piston_problem1.frames)):
        u1 = piston_problem1.frames[i].q[1]
        u2 = piston_problem2.frames[i].q[1]
        if len(indx)>0:
            u1=u1[indx]
            u2=u2[indx]
        else:
            u2 = u2[:int(len(u2)/2.)]
            u1 = u1[:len(u2)]
        max_error_absorbing = max(max_error_absorbing, error(u1,u2))
    return max_error_absorbing