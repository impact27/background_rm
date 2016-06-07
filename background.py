# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:13:14 2016

@author: quentinpeter
"""
import registration.image as ir
import numpy as np


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