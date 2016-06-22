# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:13:14 2016

@author: quentinpeter
"""
import registration.image as ir
import numpy as np
from numpy.polynomial import polynomial
import cv2
    

def match_substract(im, background):
    """Match the background to the orientation, position, scale, and intensity
    of the image and subtract it"""
    #position background on image
    angle, scale, shift, __ = ir.register_images(im,background)
    #move background
    background=ir.rotate_scale(background,angle,scale, borderValue=np.nan)
    background=ir.shift_image(background,shift, borderValue=np.nan)
    #get intensity diff
    C=match_intensity(im,background)
    #extract Data
    data=im-C*background
    
    #"""
    from matplotlib.pyplot import figure, plot
    figure()
    plot(im.mean(1))
    plot(C*np.nanmean(background,1))
    #"""
    return data

def flatten_match(im,bg, percentile=100):
    im=im/polyfit2d(im,[2,2],percentile)
    bg=bg/polyfit2d(bg,[2,2])
    #position background on image
    angle, scale, shift, __ = ir.register_images(im,bg)
    #move background
    bg=ir.rotate_scale(bg,angle,scale, borderValue=np.nan)
    bg=ir.shift_image(bg,shift, borderValue=np.nan)
    
    return im/bg
    
def match_intensity(im0,im1):
    """Matchintensity of im1 to im0"""
    #Use float32
    im0=np.asarray(im0,dtype=np.float32)
    im1=np.asarray(im1,dtype=np.float32)
    #Find valid pixels
    valid=np.logical_and(np.isfinite(im0),np.isfinite(im1))
    #computes the intensity difference
    C=(im0[valid]*im1[valid]).sum()/(im1[valid]**2).sum()
    return C
   
def polyfit2d(f, deg, percentile=100):
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
    
def polyfit2d2(f, deg, threshold=100):
    #clean input
    deg = np.asarray(deg)
    f = np.asarray(f)  
    f = cv2.GaussianBlur(f,(11,11),5)
    #get x,y
    x = np.asarray(range(f.shape[1]),dtype='float64')
    y = np.asarray(range(f.shape[0]),dtype='float64')
    X = np.array(np.meshgrid(x,y))
    
    vecs=(deg[0]+1)*(deg[1]+1)
    A=np.zeros((vecs,vecs))
    for yi in range(deg[0]+1):
        for yj in range(deg[0]+1):
            for xi in range(deg[1]+1):
                for xj in range(deg[1]+1):
                    A[(deg[1]+1)*yi+xi,(deg[1]+1)*yj+xj]=((y**(yi+yj)).sum()*
                                                          (x**(xi+xj)).sum())
    
    #get vander matrix
    vander = polynomial.polyvander2d(X[0], X[1], deg)               
    b=(vander*np.tile(f[:,:,np.newaxis],(1,1,vander.shape[2]))).sum(0).sum(0)
    c=np.linalg.solve(A, b)

    return np.dot(vander,c)
    
