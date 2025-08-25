import numpy as np
import matplotlib.pyplot as plt
#Most of this is following Toro's red book Sec 3.

def get_theta(xp, yp):
    r"""
    Get angle in radians from x and y coordinates.

    Parameters
    ----------
    xp : array_like, shape (mx,)
        x coordinates
    yp : array_like, shape (my,)
        y coordinates

    Returns
    -------
    theta : ndarray, shape (mx,my)
        Angle in [0, 2π), where theta[i,j] is the angle
        of the line from the origin to (x[i], y[j]).
    """
    X, Y = np.meshgrid(xp, yp, indexing="ij")
    theta = np.arctan2(Y, X)
    theta = np.where(theta < 0, theta + 2*np.pi, theta)
    return theta

def euclidean_distance(x1,y1,x2,y2):
    r"""
    Compute the Euclidean distance between two points (x1,y1) and (x2,y2).
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def rotate_forward(q,theta):
    r"""
    Rotate state vector of conserved quantities
    q(meqn,i,j) forward by angle wrt x-axis theta (radians).
    for each grid cell (i,j).
    theta(i,j) is the angle to rotate by.
    For 2D Euler equations, q = [rho, rho u, rho v, E]
    T = 
    \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & cos(\theta) & sin(\theta) & 0 \\
    0 & -sin(\theta) & cos(\theta) & 0 \\
    0 & 0 & 0 & 1   
    \end{bmatrix}
    """
    c = np.cos(theta)
    s = np.sin(theta)
    Tq = np.zeros_like(q)
    Tq[0,:,:] = q[0,:,:]
    Tq[1,:,:] = c*q[1,:,:] + s*q[2,:,:]
    Tq[2,:,:] = -s*q[1,:,:] + c*q[2,:,:]
    Tq[3,:,:] = q[3,:,:]
    return Tq

def rotate_backward(RTq,theta):
    r"""
    Rotate state vector of characteristic variables
    RTq (meqn,i,j) backward by angle theta wrt x-axis (radians).
    for each grid cell (i,j). 
    For 2D Euler equations, q = [rho, rho u, rho v, E]
    T^{-1} = 
    \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & cos(\theta) & -sin(\theta) & 0 \\
    0 & sin(\theta) & cos(\theta) & 0 \\
    0 & 0 & 0 & 1   
    \end{bmatrix}
    """
    c = np.cos(theta)
    s = np.sin(theta)
    q = np.zeros_like(RTq)
    q[0,:,:] = RTq[0,:,:]
    q[1,:,:] = c*RTq[1,:,:] - s*RTq[2,:,:]
    q[2,:,:] = s*RTq[1,:,:] + c*RTq[2,:,:]
    q[3,:,:] = RTq[3,:,:]
    return q

def get_1d_R(q,gamma):
    r"""
    Get eigenvector matrices for the conserved variables q in 1D (x-direction).
    """
    # Compute the eigenvalues
    rho = q[0,:,:]
    u = q[1,:,:] / rho
    v = q[2,:,:] / rho
    E = q[3,:,:]
    p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
    c = np.sqrt(gamma * p / rho)

    H = (E + p) / rho  # Total enthalpy

    # Right eigenvector matrix
    R = np.zeros((4, 4) + rho.shape)
    #First column
    R[0, 0, :, :] = 1
    R[1, 0, :, :] = u - c
    R[2, 0, :, :] = v
    R[3, 0, :, :] = H - u * c
    #Second column
    R[0, 1, :, :] = 1
    R[1, 1, :, :] = u
    R[2, 1, :, :] = v
    R[3, 1, :, :] = 0.5 * (u**2 + v**2)
    #Third column
    R[0, 2, :, :] = 0
    R[1, 2, :, :] = 0
    R[2, 2, :, :] = 1
    R[3, 2, :, :] = v
    #Fourth column
    R[0, 3, :, :] = 1
    R[1, 3, :, :] = u + c
    R[2, 3, :, :] = v
    R[3, 3, :, :] = H + u * c
    return R

def get_1d_Rinv(q,gamma):
    r"""
    Get inverse of eigenvector matrices for the conserved variables q in 1D.
    """
    # Compute the eigenvalues
    rho = q[0,:,:]
    u = q[1,:,:] / rho
    v = q[2,:,:] / rho
    E = q[3,:,:]
    p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
    c = np.sqrt(gamma * p / rho)

    beta = (gamma - 1.0) / (c**2)
    q2 = u**2 + v**2

    L = np.zeros((4, 4) + rho.shape)

    # Row 1 (acoustic -)
    L[0,0,:,:] = 0.25*beta*q2 + 0.5*u/c
    L[0,1,:,:] = (-beta*c*u - 1.0)/(2.0*c)
    L[0,2,:,:] = -0.5*beta*v
    L[0,3,:,:] = 0.5*beta

    # Row 2 (contact/entropy)
    L[1,0,:,:] = 1.0 - 0.5*beta*q2
    L[1,1,:,:] = beta*u
    L[1,2,:,:] = beta*v
    L[1,3,:,:] = -beta

    # Row 3 (shear/vorticity)
    L[2,0,:,:] = -v
    L[2,1,:,:] = 0.0
    L[2,2,:,:] = 1.0
    L[2,3,:,:] = 0.0

    # Row 4 (acoustic +)
    L[3,0,:,:] = 0.25*beta*q2 - 0.5*u/c
    L[3,1,:,:] = (-beta*c*u + 1.0)/(2.0*c)
    L[3,2,:,:] = -0.5*beta*v
    L[3,3,:,:] = 0.5*beta
    return L

def points_on_line(theta, x0=None, y0=None):
    """
    Given an array of angles theta and either scalar x0 or scalar y0,
    return an array of y (if x0 is given) or x (if y0 is given),
    such that (x0, y) or (x, y0) lies on the line through the origin at angle theta.
    
    Parameters
    ----------
    theta : array_like
        Angles (radians).
    x0 : float, optional
        Fixed x-coordinate (scalar).
    y0 : float, optional
        Fixed y-coordinate (scalar).
        
    Returns
    -------
    ndarray
        Array of y (if x0 is given) or x (if y0 is given).
    """
    theta = np.asarray(theta)

    if (x0 is None and y0 is None):
        raise ValueError("Provide one of x0 or y0 (scalars).")

    cos_t = np.cos(theta)

    if x0 is not None:
        slope = np.tan(theta)
        # return np.where(np.isclose(cos_t, 0.0),
        #                 y0,       # vertical line
        #                 slope * x0)
        #The safeguard above is not needed since we'll not  have lines crossing
        #the origin.
        return slope * x0 

    else:
        slope = np.tan(theta)
        return np.where(np.isclose(cos_t, 0.0),
                        0.0,          # vertical line => x=0
                        y0 / slope)
        
def Mayer_etal_filter(s,xL,xR,b=0.5):
    beta = lambda x: np.abs(x-xL)/np.abs(xR-xL)
    gamma = lambda x: 1-b*beta(x)**3-(1-b)*beta(x)**6
    return np.where(s<xL,1.,np.where(s>xR,0.,gamma(s)))

def compute_relax_fun(x,y,xi,yi,xf,yf):
    r"""
    s is a normalized distance along the line from (xi,yi) to (xf,yf).
    gamma(s) is 1 at s=0 and 0 at s=1.
    """
    s = euclidean_distance(x,y,xi,yi)/euclidean_distance(xi,yi,xf,yf)
    s = np.clip(s, 0, 1)  # Ensure s is within [0, 1]
    gamma = Mayer_etal_filter(s,0,1)
    return gamma

def get_sponge_idxs(xp,yp,xlims,ylims):
    r"""
    xlims contains [xLL,xL,xR,xRR]
    ylims contains [yLL,yL,yU,yUU]
    The computational domain is [xL,xR]x[yL,yU].
    The sponge layer is the disjoint union of O1,O2,O3,O4 where
    O1 = [xLL,xL]x[yLL,yUU], O2 = [xR,xRR]x[yLL,yUU],
    O3 = ]xL,xR[x[yLL,yL], O4 = ]xL,xR[x[yU,yUU].
    This routine returns:
    iL,iR,jL,jU
    where iL is the max index of xp such that xp<=xL,
    iR is the min index of xp such that xp>=xR,
    jL is the max index of yp such that yp<=yL,
    jU is the min index of yp such that yp>=yU.
    """
    xLL,xL,xR,xRR = xlims
    yLL,yL,yU,yUU = ylims
    iL = np.where(xp<=xL)[0].max()
    iR = np.where(xp>=xR)[0].min()
    jL = np.where(yp<=yL)[0].max()
    jU = np.where(yp>=yU)[0].min()
    return iL,iR,jL,jU

def get_slices_sponge_layer_general(arr,idx_limits):
    r"""
    Return 4 arrays (slices) of arr, corresponding to the 4 parts of the sponge layer.
    xlims contains [xLL,xL,xR,xRR]
    ylims contains [yLL,yL,yU,yUU]
    The computational domain is [xL,xR]x[yL,yU].
    The sponge layer is the disjoint union of O1,O2,O3,O4 where
    O1 = [xLL,xL]x[yLL,yUU], O2 = [xR,xRR]x[yLL,yUU],
    O3 = ]xL,xR[x[yLL,yL], O4 = ]xL,xR[x[yU,yUU].
    This routine returns:
    arr1,arr2,arr3,arr4
    """
    #Get indices of sponge layer
    iL,iR,jL,jU = idx_limits #get_sponge_idxs(xp,yp,xlims,ylims)
    arr1 = arr[...,:iL+1,:]  #Left
    arr2 = arr[...,iR:,:]    #Right
    arr3 = arr[...,iL+1:iR,:jL+1]  #Bottom
    arr4 = arr[...,iL+1:iR,jU:]  #Top
    return arr1,arr2,arr3,arr4

def get_slices_sponge_layer(arr,idx_limits):
    r"""
    Return 4 arrays (slices) of arr, corresponding to the 4 parts of the sponge layer.
    xlims contains [xLL,xL,xR,xRR]
    ylims contains [yLL,yL,yU,yUU]
    The computational domain is [xL,xR]x[yL,yU].
    The sponge layer is the disjoint union of O1,O2,O3,O4 where
    O1 = [xLL,xL]x[yLL,yUU], O2 = [xR,xRR]x[yLL,yUU],
    O3 = ]xL,xR[x[yLL,yL], O4 = ]xL,xR[x[yU,yUU].
    This routine returns:
    arr1,arr2,arr3,arr4
    """
    #Get indices of sponge layer
    iL,iR,jL,jU = idx_limits #get_sponge_idxs(xp,yp,xlims,ylims)
    arr1 = arr[:iL+1,:]  #Left
    arr2 = arr[iR:,:]    #Right
    arr3 = arr[iL+1:iR,:jL+1]  #Bottom
    arr4 = arr[iL+1:iR,jU:]  #Top
    return arr1,arr2,arr3,arr4


def precompute_thetas(xp,yp,idx_limits):
    theta_global = get_theta(xp,yp)
    return get_slices_sponge_layer(theta_global,idx_limits)

def precompute_relaxation_functions(xp,yp,xlims,ylims):
    r"""
    Precompute the relaxation functions for the sponge layer.
    xlims contains [xLL,xL,xR,xRR]
    ylims contains [yLL,yL,yU,yUU]
    The computational domain is [xL,xR]x[yL,yU].
    The sponge layer is the disjoint union of O1,O2,O3,O4 where
    O1 = [xLL,xL]x[yLL,yUU], O2 = [xR,xRR]x[yLL,yUU],
    O3 = ]xL,xR[x[yLL,yL], O4 = ]xL,xR[x[yU,yUU].
    This routine returns:
    rel_fun1,rel_fun2,rel_fun3,rel_fun4
    where rel_fun1 is the relaxation function for O1, etc.
    """
    #We first get the relaxation functions wrt to every side of
    #the sponge layer, then we strip to the relevant parts.
    xLL,xL,xR,xRR = xlims
    yLL,yL,yU,yUU = ylims   
    theta_global = get_theta(xp,yp)
    rel_fun_global = np.ones_like(theta_global)
    X,Y = np.meshgrid(xp,yp,indexing='ij')
    #Left side
    yi = points_on_line(theta_global,x0=xL)
    yf = points_on_line(theta_global,x0=xLL)
    xi = xL*np.ones_like(yi)
    xf = xLL*np.ones_like(yf)
    rel_fun_left = compute_relax_fun(X,Y,xi,yi,xf,yf)
    condition = np.logical_and(theta_global>=3*np.pi/4.,theta_global<=5.*np.pi/4.)
    # print("Condition left side true at this number of points:", np.sum(condition), "out of", condition.size)
    # condition = (X<xi)
    # print("Condition left side true at this number of points:", np.sum(condition), "out of", condition.size)
    rel_fun_global = np.where(condition, rel_fun_left, rel_fun_global)
    #Right side
    yi = points_on_line(theta_global,x0=xR)
    yf = points_on_line(theta_global,x0=xRR)
    xi = xR*np.ones_like(yi)
    xf = xRR*np.ones_like(yf)
    rel_fun_right = compute_relax_fun(X,Y,xi,yi,xf,yf)
    condition = np.logical_and(theta_global>=0.,theta_global<=np.pi/4.)
    condition = np.logical_or(condition, theta_global>=7.*np.pi/4.)
    # condition = (X>xi)
    rel_fun_global = np.where(condition, rel_fun_right, rel_fun_global)
    #Bottom side
    xi = points_on_line(theta_global,y0=yL)
    xf = points_on_line(theta_global,y0=yLL)
    yi = yL*np.ones_like(xi)
    yf = yLL*np.ones_like(xf)
    rel_fun_bottom = compute_relax_fun(X,Y,xi,yi,xf,yf)
    condition = np.logical_and(theta_global>=5.*np.pi/4.,theta_global<=7.*np.pi/4.)
    # condition = (Y<yi)
    rel_fun_global = np.where(condition, rel_fun_bottom, rel_fun_global)
    #Top side
    xi = points_on_line(theta_global,y0=yU)
    xf = points_on_line(theta_global,y0=yUU)
    yi = yU*np.ones_like(xi)
    yf = yUU*np.ones_like(xf)
    rel_fun_top = compute_relax_fun(X,Y,xi,yi,xf,yf)
    condition = np.logical_and(theta_global>=np.pi/4.,theta_global<=3.*np.pi/4.)
    # condition = (Y>yi)
    rel_fun_global = np.where(condition, rel_fun_top, rel_fun_global)
    #Now strip to sponge layer parts
    return get_slices_sponge_layer_general(rel_fun_global,get_sponge_idxs(xp,yp,xlims,ylims))
    # return rel_fun_global

def distance_func(X,Y,x0,y0):
    r"""
    Compute the Euclidean distance from (x0,y0) to each point in (x,y).
    x and y are 1D arrays.
    """
    return np.sqrt((X - x0)**2 + (Y - y0)**2)

    




if __name__ == "__main__":
    ########################################################
    #Verify that L is the inverse of R 
    q = np.random.rand(4,100,100)
    q[0,:,:] += 1.  # make sure density is positive
    p = 1.0
    E = p/(1.4-1.0) + 0.5*q[0,:,:]*(q[1,:,:]/q[0,:,:])**2
    q[3,:,:] = E
    gamma = 1.4
    R = get_1d_R(q,gamma)
    L = get_1d_Rinv(q,gamma)
    I = np.einsum('ik...,kj...->ij...', L, R)
    for i in range(10):
        for j in range(10):
            #Check that I is identity
            residual=np.isclose(I[:,:,i,j], np.eye(4), atol=1e-14)
            if not np.all(residual):
                print("Error: not identity")
                print(I[:,:,i,j])
                exit()
    ########################################################
    #Check points_on_line
    y0 = 0.01
    x = np.array([-1.0, 0.0, 1.0])
    y = np.ones_like(x)
    theta = get_theta(x,y)
    x_check = points_on_line(theta, y0=y0)
    # plt.scatter(x,y, label='Original points')
    # plt.scatter(x_check,y0*np.ones_like(x_check), label='Points on line x=x0')
    # plt.show()

    ########################################################
    #Check sponge layer functions
    xp = np.linspace(-6,6.,100)
    yp = np.linspace(-6,6,100)
    theta=get_theta(xp,yp)

    xlims = [-6.0,-4.0,4.0,6.0]
    ylims = [-6.0,-4.0,4.0,6.0]

    # yi = points_on_line(theta,x0=-1.5)
    # yf = points_on_line(theta,x0=-6.)

    # xi = -1.5*np.ones_like(yi)
    # xf = -6.*np.ones_like(yf)

    X,Y = np.meshgrid(xp,yp,indexing='ij')
    points = np.array([X,Y])

    O1, O2, O3, O4 = get_slices_sponge_layer_general(points,get_sponge_idxs(xp,yp,xlims,ylims))
    theta1, theta2, theta3, theta4 = get_slices_sponge_layer(theta,get_sponge_idxs(xp,yp,xlims,ylims))
    rel_fun1, rel_fun2, rel_fun3, rel_fun4 = precompute_relaxation_functions(xp,yp,xlims,ylims)


    # relax_fun = distance_func(X,Y,xf,yf)/distance_func(xi,yi,xf,yf)
    # relax_fun = distance_func(X,Y,0.0,0.0)

    # Suppose theta1..theta4 are your data arrays
    # rel_funcs = [rel_fun1, rel_fun2, rel_fun3, rel_fun4]

    # # Global min/max
    # vmin = min(np.nanmin(t) for t in rel_funcs)
    # vmax = max(np.nanmax(t) for t in rel_funcs)
    # print(vmin, vmax)

    # # Plot each block with same vmin/vmax using pcolormesh (no interpolation)
    # cs = plt.pcolormesh(O1[0], O1[1], rel_fun1, shading='nearest', vmin=vmin, vmax=vmax)
    # plt.pcolormesh(O2[0], O2[1], rel_fun2, shading='nearest', vmin=vmin, vmax=vmax)
    # plt.pcolormesh(O3[0], O3[1], rel_fun3, shading='nearest', vmin=vmin, vmax=vmax)
    # plt.pcolormesh(O4[0], O4[1], rel_fun4, shading='nearest', vmin=vmin, vmax=vmax)

    # plt.colorbar(cs)
    # plt.show()

    # Suppose theta1..theta4 are your data arrays
    rel_funcs = [rel_fun1, rel_fun2, rel_fun3, rel_fun4]
    grids = [O1, O2, O3, O4]

    # Global min/max for consistent colormap scaling
    vmin = min(np.nanmin(t) for t in rel_funcs)
    vmax = max(np.nanmax(t) for t in rel_funcs)
    print("vmin, vmax:", vmin, vmax)

    # Plot each block as scatter
    fig, ax = plt.subplots()

    for (O, rel_fun) in zip(grids, rel_funcs):
        x = O[0].ravel()
        y = O[1].ravel()
        vals = rel_fun.ravel()
        sc = ax.scatter(x, y, c=vals, cmap="viridis", vmin=vmin, vmax=vmax, s=20, marker="s")

    # Add colorbar
    plt.colorbar(sc, ax=ax)
    plt.show()

    



