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
    """
    #Remove curve from background and image (passing percentile to image)
    #The intensity has arbitraty units. To have the same variance,
    #we need to divide and not subtract
    im=im/polyfit2d(im,deg,percentile)#
    bg=bg/polyfit2d(bg,deg)
    
    angle=None
    if xOrientate:    
        angle=ir.orientation_angle(im)
    
    #position background on image
    angle, scale, shift, __ = ir.register_images(im,bg)
    #move background
    bg=ir.rotate_scale(bg,angle,scale, borderValue=np.nan)
    bg=ir.shift_image(bg,shift, borderValue=np.nan)
    
    #resize if shape is not equal
    if im.shape is not bg.shape:
        im, bg = same_size(im,bg)
    #rotate result
    data=im-bg
    #if we want to orientate the image, do it now
    if xOrientate:
        #rotate
        data=ir.rotate_scale(data,-angle,1, borderValue=np.nan)
    
    """
    from matplotlib.pyplot import figure, plot
    figure()
    plot(np.nanmean(im,1))
    plot(np.nanmean(bg,1))
    #"""
    #subtract the image
    return data
    
def same_size(im0,im1):    
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
    valid=f<np.percentile(f,percentile)
    #Find coefficients
    c = np.linalg.lstsq(vander[valid,:], f[valid])[0]
    #Compute value
    ret=np.dot(vander,c)
    return ret.reshape(initshape)
   
#to use this one, we need to solve the case of a non x-orientated image
def polyfit2d2(f, deg, yPercentile=100):
    """Fit the function f to the degree deg
    Ignore everithing above yPercentile (mean, ...)
    """
    #clean input
    deg = np.asarray(deg)
    f = np.asarray(f)  
    #f = cv2.GaussianBlur(f,(11,11),5)
    #get x,y
    x = np.asarray(range(f.shape[1]),dtype='float64')
    y = np.asarray(range(f.shape[0]),dtype='float64')
    X = np.array(np.meshgrid(x,y))
    
    ymean=f.mean(1)
    yvalid=ymean< np.percentile(ymean,yPercentile)
    
    vecs=(deg[0]+1)*(deg[1]+1)
    A=np.zeros((vecs,vecs))
    for yi in range(deg[0]+1):
        for yj in range(deg[0]+1):
            for xi in range(deg[1]+1):
                for xj in range(deg[1]+1):
                    A[(deg[1]+1)*yi+xi,(deg[1]+1)*yj+xj]=(
                        (y[yvalid]**(yi+yj)).sum()*(x**(xi+xj)).sum())
    
    #get vander matrix
    vander = polynomial.polyvander2d(X[1], X[0], deg)    
    b=(vander[yvalid,:,:]*np.tile(f[yvalid,:,np.newaxis],
            (1,1,vander.shape[2]))).sum(0).sum(0)
    c=np.linalg.solve(A, b)
    return np.dot(vander,c)
 
#old version
#   
#def match_substract(im, background):
#    """Match the background to the orientation, position, scale, and intensity
#    of the image and subtract it"""
#    #position background on image
#    angle, scale, shift, __ = ir.register_images(im,background)
#    #move background
#    background=ir.rotate_scale(background,angle,scale, borderValue=np.nan)
#    background=ir.shift_image(background,shift, borderValue=np.nan)
#    #get intensity diff
#    C=match_intensity(im,background)
#    #extract Data
#    data=im-(C*background)
#    
#    """
#    from matplotlib.pyplot import figure, plot
#    figure()
#    plot(im.mean(1))
#    plot(C*np.nanmean(background,1))
#    #"""
#    return data
#    
#def match_intensity(im0,im1):
#    """Matchintensity of im1 to im0"""
#    #Use float32
#    im0=np.asarray(im0,dtype=np.float32)
#    im1=np.asarray(im1,dtype=np.float32)
#    #Find valid pixels
#    valid=np.logical_and(np.isfinite(im0),np.isfinite(im1))
#    #computes the intensity difference
#    C=(im0[valid]*im1[valid]).sum()/(im1[valid]**2).sum()
#    return C