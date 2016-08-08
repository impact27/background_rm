# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:13:14 2016

@author: quentinpeter

This scripts automates the background removal

Copyright (C) 2016  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import image_registration.image as ir
import numpy as np
from numpy.polynomial import polynomial
import cv2
from scipy.special import erfinv

def remove_curve_background(im, bg, percentile=None, deg=[2,2], *, 
                            xOrientate=False, twoPass=False,threePass=False):
    """flatten the image by removing the curve and the background fluorescence. 
    
    Parameters
    ----------
    im: 2d array
        The image with background and data
    bg: 2d array
        The imabge with only background
    percentile: number 0-100, optional
        The percentage of the image covered by the background
        If None, the script uses morphological functions to find the proteins
    deg: 2 numbers, default [2,2]
        The polynomial fit Y and X degrees
    xOrientate: boolean, default False
        if True, will orientate the image along the x axis
    twoPass: boolean, defaults False
        Uses 2 pass to get flatter result
    
    Returns
    -------
    im: 2d array
        The image with background removed
        
    Notes
    -----
    TODO: this function takes almost 2 seconds for a single pair of 1000x1000
    matrices!!! 1.2 of which for folyfit2d! need to optimize that
    """
    #Remove curve from background and image (passing percentile to image)
    #The intensity has arbitraty units. To have the same variance,
    #we need to divide and not subtract
    im=im/polyfit2d(im,deg,percentile)#
    bg=bg/polyfit2d(bg,deg,100)
    
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
    
    #If 2 pass, get flatter image and resubtract
    if twoPass:
        mask=getValid(cv2.GaussianBlur(data,(21,21),0))
        im=im/polyfit2d(im,deg,mask=mask)
        data=im-bg
        
    if threePass:
        mask=getValid(cv2.GaussianBlur(data,(21,21),0))
        im=im/polyfit2d(im,deg,mask=mask)
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
    
    Parameters
    ----------
    im0: 2d array
        First image
    im1: 2d array
        Second image
        
    Returns
    -------
    im0: 2d array
        First image
    im1: 2d array
        Second image
        
    """
    shape=[max(im0.shape[0],im1.shape[0]),max(im0.shape[1],im1.shape[1])]
    im0 = cv2.copyMakeBorder(im0, 0, shape[0] - im0.shape[0],
                                0, shape[1] - im0.shape[1], 
                                borderType=cv2.BORDER_CONSTANT, value=np.nan)
    im1 = cv2.copyMakeBorder(im1, 0, shape[0] - im1.shape[0],
                                0, shape[1] - im1.shape[1], 
                                borderType=cv2.BORDER_CONSTANT, value=np.nan)
    
    return im0,im1
   
def getValid(im, r=2):
    
    #take mode
    immin=np.nanmin(im)
    immax=np.nanmax(im)
    #hist = cv2.calcHist([f[np.isfinite(f)]],[0],None,[1000],[fmin,fmax])
    hist,*_= np.histogram(im[np.isfinite(im)],1000,[immin,immax])
    m=hist[1:-1].argmax()+1#don't want saturated values
    hm=m
    m=m*(immax-immin)/1000+immin
    #if hist 0 is saturated, use erfinv
    if hist[0]>0.5*hist.max():
        std=(-m)/np.sqrt(2)/erfinv(hist[0]/hist[:hm].sum()-1)
    else:
        #std everithing below mean
        std=np.sqrt(((m-im[im<m])**2).mean())
    #3 std should be good
    valid=im<m+3*std

    def turad(r):
        return (2*r+1,2*r+1)
    #remove dots in proteins (3px dots)
    valid=cv2.erode(np.asarray(valid,dtype="uint8"),np.ones(turad(r)))
    #remove dots in background (2 px dots)
    valid=cv2.dilate(valid,np.ones(turad(r+1)))
    #widen proteins (10 px around proteins)
    valid=cv2.erode(valid,np.ones(turad(r+1)))

    #If invalid values in im, get rid of them
    valid=np.logical_and(valid,np.isfinite(im))
    
    """
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(valid)
    #"""
    return valid
   
def polyfit2d(im, deg=[2,2], Percentile=None, mask=None):
    """Fit the function f to the degree deg
    
    Parameters
    ----------
    im: 2d array
        The image to fit
    deg: 2 numbers, defaults [2,2]
        The Y and X polynomial degrees to fit
    Percentile: number, optional
        The percentage of the image covered by the background
        If None, the script uses morphological functions to find the proteins
    
    Returns
    -------
    im: 2d array
        The fitted polynomial surface
        
    Notes
    -----
    Ignore everithing above Percentile (mean, ...)
    """
    #clean input
    deg = np.asarray(deg)
    im = np.asarray(im,dtype='float32')  
    
    
    #get x,y
    x = np.asarray(range(im.shape[1]),dtype='float32')[np.newaxis,:]
    y = np.asarray(range(im.shape[0]),dtype='float32')[:,np.newaxis]
    
    if mask is not None:
        valid=mask
                     
    elif Percentile is not None:
        im = cv2.GaussianBlur(im,(11,11),0)
        #compare percentile with blured version
        valid=im< np.nanpercentile(im,Percentile)
        
    else:
        #im = cv2.GaussianBlur(im,(3,3),0)
        valid=getValid(im)
        
    
    #"""
    if Percentile is not 100:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(valid)
    #"""
    
    vecs=(deg[0]+1)*(deg[1]+1)
    
    res=np.zeros(((deg[0]*2+1),(deg[1]*2+1)),dtype='float32')
    vander=np.zeros((vecs,*(im.shape)),dtype='float32')
    vandervalid=vander.copy()
    vm=np.zeros(im.shape,dtype='float32')
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
    
    d=im.copy()
    d[np.logical_not(valid)]=0
    b=np.dot(np.reshape(vandervalid,(vandervalid.shape[0],-1)),
             np.reshape(d,(-1,)))
    c=np.linalg.solve(A, b)
    return np.dot(np.moveaxis(vander,0,-1),c)
    
def polyfit2d_alt(im, deg=[2,2], percentile=100):
    """Fit the function f to the degree deg
    
    Parameters
    ----------
    im: 2d array
        The image to fit
    deg: 2 numbers, defaults [2,2]
        The Y and X polynomial degrees to fit
    Percentile: number, optional
        The percentage of the image covered by the background
    
    Returns
    -------
    im: 2d array
        The fitted polynomial surface
        
    Notes
    -----
    Ignore everithing above percentile
    This is kind of wrong as y^2 * x^2 is not 2nd degree...
    """
    #clean input
    deg = np.asarray(deg)
    im = np.asarray(im)  
    im = cv2.GaussianBlur(im,(11,11),0)
    #get x,y
    x = np.asarray(range(im.shape[1]))
    y = np.asarray(range(im.shape[0]))
    X = np.array(np.meshgrid(x,y))
    #save shape
    initshape=im.shape
    #get vander matrix
    vander = polynomial.polyvander2d(X[0], X[1], deg)
    #reshape for lstsq
    vander = vander.reshape((-1,vander.shape[-1]))
    im = im.reshape((vander.shape[0],))
    #get valid values
    valid=im<np.nanpercentile(im,percentile)
    #Find coefficients
    c = np.linalg.lstsq(vander[valid,:], im[valid])[0]
    #Compute value
    ret=np.dot(vander,c)
    return ret.reshape(initshape)

    
    #    vandermonde=np.zeros(((deg[0]*2+1),(deg[1]*2+1),*(im.shape)),dtype='float64')
#    for yp in range(deg[0]*2+1):
#        for xp in range(deg[1]*2+1):
#            #There is no clear need to recompute that each time
#            np.dot((y**yp),(x**xp),out=vandermonde[yp,xp,:,:])
#          
#    vanmonvalid=np.dot(vandermonde.reshape((*(vandermonde.shape[:2]),-1)),
#                                              valid.reshape((-1,)))
#    res=vanmonvalid.sum(-1)
