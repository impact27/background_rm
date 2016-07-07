# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:13:14 2016

@author: quentinpeter
"""
import image_registration.image as ir
import numpy as np
from numpy.polynomial import polynomial
import cv2

def remove_curve_background(im, bg, percentile=100, *, xOrientate=False,
                                                                    deg=[2,2]):
    """flatten the image by removing the curve and the background fluorescence. 
    Need to have an image of the background
    TODO: this function takes almost 2 seconds for a single pair of 1000x1000
    matrices!!! 1.2 of which for folyfit2d! need to optimize that
    """
    #Remove curve from background and image (passing percentile to image)
    #The intensity has arbitraty units. To have the same variance,
    #we need to divide and not subtract
    im=im/polyfit2d2(im,deg,percentile)#
    bg=bg/polyfit2d2(bg,deg)
    
    #if the image or the background have any nans, replace by 1 (for fft)
    nanim=np.isnan(im)
    nanbg=np.isnan(bg)
    im[nanim]=1
    bg[nanbg]=1
       
    #make FFT calculations on background and image
    if xOrientate:    
        angleOri=ir.orientation_angle(im)
    angle, scale, shift, __ = ir.register_images(im,bg)
    
    #remobe the previously added nans
    im[nanim]=np.nan
    bg[nanbg]=np.nan
    
    #move background
    bg=ir.rotate_scale(bg,angle,scale, borderValue=np.nan)
    bg=ir.shift_image(bg,shift, borderValue=np.nan)
    
    #resize if shape is not equal
    if im.shape is not bg.shape:
        im, bg = same_size(im,bg)
        
    #subtract background
    data=im-bg
    
    #if we want to orientate the image, do it now
    if xOrientate:
        #rotate
        data=ir.rotate_scale(data,-angleOri,1, borderValue=np.nan)
    
    """
    from matplotlib.pyplot import figure, plot
    figure()
    plot(np.nanmean(im,1))
    plot(np.nanmean(bg,1))
    #"""
    #return result
    return data
    
def same_size(im0,im1):    
    """Pad with nans to get similarely shaped matrix
    """
    shape=[max(im0.shape[0],im1.shape[0]),max(im0.shape[1],im1.shape[1])]
    im0 = cv2.copyMakeBorder(im0, 0, shape[0] - im0.shape[0],
                                0, shape[1] - im0.shape[1], 
                                borderType=cv2.BORDER_CONSTANT, value=np.nan)
    im1 = cv2.copyMakeBorder(im1, 0, shape[0] - im1.shape[0],
                                0, shape[1] - im1.shape[1], 
                                borderType=cv2.BORDER_CONSTANT, value=np.nan)
    
    return im0,im1
   
def polyfit2d(f, deg, percentile=100):
    """Fit the function f to the degree deg
    Ignore everithing above percentile
    This is kind of wrong as y^2 * x^2 is not 2nd degree...
    """
    #clean input
    deg = np.asarray(deg)
    f = np.asarray(f)  
    f = cv2.GaussianBlur(f,(11,11),5)
    #get x,y
    x = np.asarray(range(f.shape[1]))
    y = np.asarray(range(f.shape[0]))
    X = np.array(np.meshgrid(x,y))
    #save shape
    initshape=f.shape
    #get vander matrix
    vander = polynomial.polyvander2d(X[0], X[1], deg)
    #reshape for lstsq
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    #get valid values
    valid=f<np.nanpercentile(f,percentile)
    #Find coefficients
    c = np.linalg.lstsq(vander[valid,:], f[valid])[0]
    #Compute value
    ret=np.dot(vander,c)
    return ret.reshape(initshape)
   
def polyfit2d2(f, deg, Percentile=100):
    """Fit the function f to the degree deg
    Ignore everithing above Percentile (mean, ...)
    """
    #clean input
    deg = np.asarray(deg)
    f = np.asarray(f,dtype='float32')  
    f = cv2.GaussianBlur(f,(11,11),5)
    #get x,y
    x = np.asarray(range(f.shape[1]),dtype='float32')[np.newaxis,:]
    y = np.asarray(range(f.shape[0]),dtype='float32')[:,np.newaxis]
    
    valid=f< np.nanpercentile(f,Percentile)
    
    vecs=(deg[0]+1)*(deg[1]+1)
    
    res=np.zeros(((deg[0]*2+1),(deg[1]*2+1)),dtype='float32')
    vander=np.zeros((vecs,*(f.shape)),dtype='float32')
    vandervalid=vander.copy()
    vm=np.zeros(f.shape,dtype='float32')
    vmvalid=vm.copy()
    for yp in range(deg[0]*2+1):
        for xp in range(deg[1]*2+1):
            #There is no clear need to recompute that each time
            np.dot((y**yp),(x**xp),out=vm)
            np.multiply(vm,valid,out=vmvalid)
            res[yp,xp]=(vmvalid).sum()
            if yp<deg[0]+1 and xp <deg[1]+1:
                vander[(deg[0]+1)*yp+xp,:,:]=vm
                vandervalid[(deg[0]+1)*yp+xp,:,:]=vmvalid
    
    A=np.zeros((vecs,vecs),dtype='float64')
    for yi in range(deg[0]+1):
        for yj in range(deg[0]+1):
            for xi in range(deg[1]+1):
                for xj in range(deg[1]+1):
                    A[(deg[1]+1)*yi+xi,(deg[1]+1)*yj+xj]=res[yi+yj,xi+xj]
    
    d=f.copy()
    d[np.logical_not(valid)]=0
    b=np.dot(np.reshape(vandervalid,(vandervalid.shape[0],-1)),np.reshape(d,(-1,)))
    c=np.linalg.solve(A, b)
    return np.dot(np.moveaxis(vander,0,-1),c)
    
    #    vandermonde=np.zeros(((deg[0]*2+1),(deg[1]*2+1),*(f.shape)),dtype='float64')
#    for yp in range(deg[0]*2+1):
#        for xp in range(deg[1]*2+1):
#            #There is no clear need to recompute that each time
#            np.dot((y**yp),(x**xp),out=vandermonde[yp,xp,:,:])
#          
#    vanmonvalid=np.dot(vandermonde.reshape((*(vandermonde.shape[:2]),-1)),
#                                              valid.reshape((-1,)))
#    res=vanmonvalid.sum(-1)
