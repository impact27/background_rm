# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 08:15:07 2017

@author: quentinpeter
"""
import numpy as np
from numba.pycc import CC

cc = CC('polyfitnumba')
cc.verbose = True


#@cc.export('polyfit2dcoeff',
#           'f4[:](f4[:, :], f4[:, :], b1[:, :], f4[:, :], f4[:, :], i8)')
#def polyfit2dcoeff(im, vander, valid, x, y, deg=2):
#    """Fit the function f to the degree deg
#
#    Parameters
#    ----------
#    im: 2d array
#        The image to fit
#    x: 1d array
#        X position
#    y: 1d array
#        Y positions
#    deg: integer, default 2
#        The polynomial degree to fit
#    mask: 2d boolean array
#        Alternative to percentile, valid values
#    vanderOut: 2d array
#        Outpts vander matrix if requested
#
#    Returns
#    -------
#    im: 2d array
#        The fitted polynomial surface
#
#    Notes
#    -----
#    To do a least square of the image, we need to minimize sumOverPixels (SOP)
#    ((fit-image)**2)
#    Where fit is a function of C_ij:
#        fit=sum_ij(C_ij * y**i * x**j)
#
#    we define the Vandermonde matrix as follow:
#        V[i,j]=y**i * x**j
#
#    where x and y are meshgrid of the y and x index
#
#    So we want the derivate of SOP((fit-image)**2) relative to the C_ij
#    to be 0.
#
#        d/dCij SOP((fit-image)**2) = 0 = 2*SOP((fit-image)*d/dC_ij fit)
#
#    Therefore
#
#        SOP(sum_kl(C_kl * V[k+i,l+j]))=SOP(image*V[i,j])
#        sum_kl(C_kl * SOP(V[k+i,l+j]))=SOP(image*V[i,j])
#
#    Which is a simple matrix equation! A*C=B with A.size=(I+J)*(I+J),
#    C.size=B.size=I+J
#
#    The sizes of the matrices are only of the order of deg**4
#
#    The bottleneck is any operation on the images before the SOP
#    """
#
#    
#
#    # Number of x and y power combinations
#    psize = ((deg + 1) * (deg + 2)) // 2
#    # This will hold the sum of the vandermonde matrix,
#    # square instead of shape(psize) for readibility.
#    # This will therefore be a upper triangular matrix
#    SOPV = np.empty(((deg * 2 + 1), (deg * 2 + 1)), dtype='float32')
#    # vandermonde matrix with all masked values =0
#    vandermasked = np.empty_like(vander)
#    # Temp. matrix that will hold the current value of vandermonde
#    vtmp = np.empty_like(im)
#    # idem but with 0 on masked pixels
#    vtmpmasked = np.empty_like(im)
#    
#    
#    tightinnerloop(im, vander, valid, x, y, deg, vtmp, vtmpmasked, SOPV, 
#                   vandermasked)
#    # Then compute the LHS of the least square equation
#    A = np.zeros((psize, psize), dtype='float32')
#    for yi in range(deg + 1):
#        for yj in range(deg + 1):
#            for xi in range(deg + 1 - yi):
#                for xj in range(deg + 1 - yj):
#                    A[getidx(yi, xi, deg), getidx(yj, xj, deg)
#                      ] = SOPV[yi + yj, xi + xj]
#
#    # Set everithing invalid to 0 (as x*0 =0 works for any x)
#    if valid is not None:
#        d = im.copy()
#        d[np.logical_not(valid)] = 0
#    else:
#        d = im
#        vandermasked = vander
#
#    # Get the RHS of the least square equation
#    b = np.dot(np.reshape(vandermasked, (vandermasked.shape[0], -1)),
#               np.reshape(d, (-1,)))
#
#    # Solve
#    c = np.linalg.solve(A, b)
#
#    """
#    if valid is None:
#        valid=[[0,0],[0,0]]
#    from matplotlib.pyplot import figure, imshow
#    figure()
#    imshow(valid)
#    #"""
#
#    return c



@cc.export('tightinnerloop',
           'i8(f4[:, :], f4[:, :], b1[:, :], f4[:, :], f4[:, :], i8, '
           'f4[:, :], f4[:, :], f4[:, :], f4[:, :])')
def tightinnerloop(im, vander, valid, x, y, deg, vtmp, vtmpmasked, SOPV, 
                   vandermasked):

    # First thing is to compute the vandermonde matrix
    for yp in range(deg * 2 + 1):
        for xp in range(deg * 2 + 1 - yp):
            # There is no clear need to recompute that each time
            np.dot((y**yp), (x**xp), out=vtmp)
            np.multiply(vtmp, valid, out=vtmpmasked)
            SOPV[yp, xp] = vtmpmasked.sum()

            if yp < deg + 1 and xp < deg + 1 - yp:
                idx = ((2 * (deg + 1) + 1) * yp - yp**2) // 2 + xp
                vander[idx] = vtmp
                vandermasked[idx] = vtmpmasked
    return 0


    


if __name__ == "__main__":
    cc.compile()
