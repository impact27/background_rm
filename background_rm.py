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
import image_registration.channel as cr
import numpy as np
import cv2
from scipy.special import erfinv
import warnings
import scipy
sobel = scipy.ndimage.filters.sobel

def remove_curve_background(im, bg, percentile=None, deg=2, *, 
                            xOrientate=False, method='none', infoDict=None):
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
        Uses 2 pass to get flatter result. Do not use with detectChannel
    detectChannel: boolean, default False
        Tries to detect the channel.
    infoDict: dictionnary
        If not None, will contain the infos on modification
    
    Returns
    -------
    im: 2d array
        The image with background removed
        
    Notes
    -----
    for detectChannel, The channel should be clearly visible and straight.
    TODO: this function takes almost 2 seconds for a single pair of 1000x1000
    matrices!!! 1.2 of which for folyfit2d! need to optimize that
    """
    
    #create mask to detect the background
    mask=None
    if percentile is None:
        mask=backgroundMask(im)
      
    #Flatten the image 
    im=im/polyfit2d(im,deg,percentile, mask=mask)#
    #Consider the background is flat (remove obvious dust)
    bg=bg/polyfit2d(bg,deg,mask=backgroundMask(im, nstd=6))
    
    #if the image has any nans, replace by 1 (for fft)
    nanim=np.isnan(im)
    nanbg=np.isnan(bg)
    im[nanim]=1
    bg[nanbg]=1

    #Detect the image angle if needed
    angleOri=0
    if xOrientate or method == 'detectChannel' or method == 'gaussianBeam':    
        angleOri=ir.orientation_angle(im)    
     
    #get angle scale and shift
    angle, scale, shift, __ = ir.register_images(im,bg)
    
    #remove the previously added nans
    im[nanim]=np.nan
    bg[nanbg]=np.nan
    
    #move background
    bg=ir.rotate_scale_shift(bg,angle,scale,shift, borderValue=np.nan)
    
    #resize if shape is not equal
    if im.shape is not bg.shape:
        im, bg = same_size(im,bg)
        
    #If detect channel, correct with channel and proceed
    if method == 'detectChannel':
        mask= outChannelMask(bg,angleOri)
        if mask is not None:
            im=im/polyfit2d(im,deg,mask=mask)
            bg=bg/polyfit2d(bg,deg,mask=mask)   
        
    #subtract background
    data=im-bg
        
    #If 2 pass, get flatter image and resubtract    
    if method == 'twoPass':
        mask=backgroundMask(data,im.shape[0]//100,blur=True)
        im=im/polyfit2d(im,deg,mask=mask)
        #bg=bg/polyfit2d(bg,deg,mask=mask)
        data=im-bg
    elif method == 'gaussianBeam':
        mask= outGaussianBeamMask(data,angleOri)
        if mask is not None:
            im=im/polyfit2d(im,deg,mask=mask)
            bg=bg/polyfit2d(bg,deg,mask=mask)
            data=im-bg
               
    #if we want to orientate the image, do it now
    if xOrientate:
        #rotate
        data=ir.rotate_scale(data,-angleOri,1, borderValue=np.nan)
        
    if infoDict is not None:
        infoDict['imageAngle']=angleOri
        infoDict['diffAngle']=angle
        infoDict['diffScale']=scale
        infoDict['offset']=shift
    
   
    """
    from matplotlib.pyplot import figure, plot
    im[np.isnan(data)]=np.nan
    bg[np.isnan(data)]=np.nan
    im=ir.rotate_scale(im,-angleOri,1, borderValue=np.nan)
    bg=ir.rotate_scale(bg,-angleOri,1, borderValue=np.nan)
    figure()
    plot(np.nanmean(im,1))
    plot(np.nanmean(bg,1))
    plot(np.nanmean(data,1)+1)
    plot([0,im.shape[0]],[1,1])
    #"""
    
    #return result
    return data
 
def outChannelMask(im, chAngle=0):
    """Creates a mask that excludes the channel
    
    Parameters
    ----------
    im: 2d array
        The image
    chAngle: number
        The angle of the channel in radians
    
    Returns
    -------
    mask: 2d array
        the mask excluding the channel
        
    Notes
    -----
    The channel should be clear(ish) on the image. 
    The angle should be aligned with the channel
    

    """
    im=np.array(im,dtype='float32')
    #Remove clear dust
    mask=backgroundMask(im, nstd=6)
    im[~mask]=np.nan
    
    #get edge
    scharr=cr.Scharr_edge(im)
    #Orientate image along x if not done
    if chAngle !=0:
        scharr= ir.rotate_scale(scharr, -chAngle,1,np.nan)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
            #get profile
        prof=np.nanmean(scharr,1)
        #get threshold
        threshold=np.nanmean(prof)+3*np.nanstd(prof)
        mprof=prof>threshold
        edgeargs=np.flatnonzero(mprof)
        
        if edgeargs.size > 2:
            mask=np.zeros(im.shape)
            mask[edgeargs[0]-5:edgeargs[-1]+5,:]=2
            if chAngle !=0:
                mask= ir.rotate_scale(mask, chAngle,1,np.nan)
            mask=np.logical_and(mask<1, np.isfinite(im))
        else:
            mask= None
    return mask
    
def outGaussianBeamMask(data, chAngle=0):
    """
    A single, straight, protein beam is present. It is "Sinking" the profile 
    such as the sides are leaning toward the center
    """
    data=np.asarray(data)
    
    #Filter to be used
    gfilter=scipy.ndimage.filters.gaussian_filter1d
    
    #get profile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  
        profile=np.nanmean(ir.rotate_scale(data, -chAngle,1,np.nan),1)
    
    #guess position of max
    amax= profile.size//2
    
    #get X and Y
    X0=np.arange(profile.size)-amax
    Y0=profile
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        #The cutting values are when the profiles goes below zero
        rlim=np.flatnonzero(np.logical_and(Y0<0,X0>0))[0]
        llim=np.flatnonzero(np.logical_and(Y0<0,X0<0))[-1]
    
    #We can now detect the true center
    fil=gfilter(profile,21)
    X0=X0-X0[np.nanargmax(fil[llim:rlim])]-llim
    
    #restrict to the correct limits
    X=X0[llim:rlim]
    Y=Y0[llim:rlim]-np.nanmin(Y0)
    
    #Fit the log, which should be a parabola
    c=np.polyfit(X,np.log(Y),2)
    
    #Deduce the variance
    var=-1/(2*c[0])
    
    #compute the limits (3std, restricted to half the image)
    mean=np.nanargmax(fil[llim:rlim])+llim
    dist=int(3*np.sqrt(var))
    if dist > profile.size//4:
        dist = profile.size//4
    llim=mean-dist
    if llim < 0:
        return None
    rlim=mean+dist
    if rlim>profile.size:
        return None
    
    #get mask
    mask=np.ones(data.shape)
    mask[llim:rlim,:]=0
    mask= ir.rotate_scale(mask, chAngle,1,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        mask=np.logical_and(mask>.5, np.isfinite(data))
    return mask
    
    """
    import matplotlib.pyplot as plt
    plt.figure()
    #plot profile and fit
    valmax=np.nanmax(Y)
    plt.plot(X0,Y0)
    plt.plot(X0,valmax*np.exp(-(X0**2)/(2*var))+np.nanmin(Y0))
    plt.plot([llim-mean,llim-mean],[np.nanmin(Y0),np.nanmax(Y0)],'r')
    plt.plot([rlim-mean,rlim-mean],[np.nanmin(Y0),np.nanmax(Y0)],'r')
    #"""
    
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

    
def backgroundTreshold(im, nstd=3):
    """get a threshold to remove background
    
    Parameters
    ----------
    im: 2d array
        The image
        
    Returns
    -------
    threshold: number
        the threshold
    
    """
    #take mode
    
    immin=np.nanmin(im)
    immax=np.nanmax(im)
    hist,*_= np.histogram(im[np.isfinite(im)],1000,[immin,immax])
    m=hist[1:-1].argmax()+1#don't want saturated values
    hm=m
    m=m*(immax-immin)/1000+immin
    #if hist 0 is saturated, use erfinv
    if hist[0]>0.5*hist.max():
        std=(-m)/np.sqrt(2)/erfinv(hist[0]/hist[:hm].sum()-1)
    else:
        #std everithing below mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            std=np.sqrt(((m-im[im<m])**2).mean())
    #3 std should be good
    return m+nstd*std
 
def getCircle(r):
    """Return round kernel for morphological operations
    
    Parameters
    ----------
    r: uint
        The radius
        
    Returns
    -------
    ker: 2d array
        the kernel
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
    
def backgroundMask(im, r=2, blur=False, nstd=3):
    """Tries to extract the background of the image
    
    Parameters
    ----------
    im: 2d array
        The image
    r: uint
        The radius used to fill the gaps
        
    Returns
    -------
    valid: 2d array
        the valid mask
    
    """
    finite=np.isfinite(im)
    if blur:
        im=cv2.GaussianBlur(im,(2*r+1,2*r+1),0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid=im<backgroundTreshold(im,nstd)
    valid=np.asarray(valid,dtype="uint8")
    
    #remove dots in proteins (3px dots)
    valid=cv2.erode(valid,getCircle(r))
    #remove dots in background (2 px dots)
    valid=cv2.dilate(valid,getCircle(r+r//2))
    #widen proteins (10 px around proteins)
    valid=cv2.erode(valid,getCircle(r))

    #If invalid values in im, get rid of them
    valid=np.logical_and(valid,finite)
    
    return valid
    
def signalMask(im, r=2, blur=False):
    """Tries to extract the signal of the image
    
    Parameters
    ----------
    im: 2d array
        The image
    r: uint
        The radius used to fill the gaps
        
    Returns
    -------
    valid: 2d array
        the valid mask
    
    """
    finite=np.isfinite(im)
    if blur:
        im=cv2.GaussianBlur(im,(2*r+1,2*r+1),0)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid=im>backgroundTreshold(im)
    valid=np.asarray(valid,dtype="uint8")

    #remove dots in proteins (3px dots)
    valid=cv2.dilate(valid,getCircle(r))
    #remove dots in background (2 px dots)
    valid=cv2.erode(valid,getCircle(r+r//2))
    #widen proteins (10 px around proteins)
    valid=cv2.dilate(valid,getCircle(r//2))

    #If invalid values in im, get rid of them
    valid=np.logical_and(valid,finite)
    
    return valid
 
def polyfit2d(im, deg=2, percentile=None, mask=None):
    """Fit the function f to the degree deg
    
    Parameters
    ----------
    im: 2d array
        The image to fit
    deg: integer, default 2
        The polynomial degree to fit
    percentile: number (0-100), optional
        The percentage of the image covered by the background
    mask: 2d boolean array
        Alternative to percentile, valid values
    
    Returns
    -------
    im: 2d array
        The fitted polynomial surface
        
    Notes
    -----
    To do a least square of the image, we need to minimize sumOverPixels (SOP)
    ((fit-image)**2)
    Where fit is a function of C_ij:
        fit=sum_ij(C_ij * y**i * x**j)
       
    we define the Vandermonde matrix as follow:
        V[i,j]=y**i * x**j
        
    where x and y are meshgrid of the y and x index
    
    So we want the derivate of SOP((fit-image)**2) relative to the C_ij
    to be 0.
    
        d/dCij SOP((fit-image)**2) = 0 = 2*SOP((fit-image)*d/dC_ij fit)
    
    Therefore
    
        SOP(sum_kl(C_kl * V[k+i,l+j]))=SOP(image*V[i,j])
        sum_kl(C_kl * SOP(V[k+i,l+j]))=SOP(image*V[i,j])
    
    Which is a simple matrix equation! A*C=B with A.size=(I+J)*(I+J),
    C.size=B.size=I+J
        
    The sizes of the matrices are only of the order of deg**4
    
    The bottleneck is any operation on the images before the SOP 
    """
    #clean input
    im = np.asarray(im,dtype='float32')  
    
    #get x,y
    x = np.asarray(range(im.shape[1]),dtype='float32')[np.newaxis,:]
    y = np.asarray(range(im.shape[0]),dtype='float32')[:,np.newaxis]
    
    valid=np.isfinite(im)
    if mask is not None:
        valid=np.logical_and(valid,mask)
                     
    elif percentile is not None and percentile is not 100:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid=im< np.nanpercentile(im,percentile)
        
    if np.all(valid):
        valid=None
        
    
    #Number of x and y power combinations
    psize=((deg+1)*(deg+2))//2
    #This will hold the sum of the vandermonde matrix, 
    #square instead of shape(psize) for readibility.
    #This will therefore be a upper triangular matrix
    SOPV=np.empty(((deg*2+1),(deg*2+1)),dtype='float32')
    #vandermonde matrix
    vander=np.empty((psize,*(im.shape)),dtype='float32')
    #vandermonde matrix with all masked values =0
    vandermasked=np.empty((psize,*(im.shape)),dtype='float32')
    #Temp. matrix that will hold the current value of vandermonde
    vtmp=np.empty(im.shape,dtype='float32')
    #idem but with 0 on masked pixels
    vtmpmasked=np.empty(im.shape,dtype='float32')
    
    #function to order powers in psize
    def getidx(y,x):
        return ((2*(deg+1)+1)*y-y**2)//2+x

    #First thing is to compute the vandermonde matrix
    for yp in range(deg*2+1):
        for xp in range(deg*2+1-yp):
            #There is no clear need to recompute that each time
            np.dot((y**yp),(x**xp),out=vtmp)
            if valid is not None:
                np.multiply(vtmp,valid,out=vtmpmasked)
                SOPV[yp,xp]=vtmpmasked.sum()
            else:
                SOPV[yp,xp]=vtmp.sum()
            
            if yp<deg+1 and xp <deg+1-yp:
                vander[getidx(yp,xp),:,:]=vtmp
                if valid is not None:
                    vandermasked[getidx(yp,xp),:,:]=vtmpmasked
    
    #Then compute the LHS of the least square equation
    A=np.zeros((psize,psize),dtype='float32')
    for yi in range(deg+1):
        for yj in range(deg+1):
            for xi in range(deg+1-yi):
                for xj in range(deg+1-yj):
                    A[getidx(yi,xi),getidx(yj,xj)]=SOPV[yi+yj,xi+xj]
    
    #Set everithing invalid to 0 (as x*0 =0 works for any x)
    if valid is not None:
        d=im.copy()
        d[np.logical_not(valid)]=0
    else:
        d=im
        vandermasked=vander

    #Get the RHS of the least square equation
    b=np.dot(np.reshape(vandermasked,(vandermasked.shape[0],-1)),
             np.reshape(d,(-1,)))
    
    #Solve
    c=np.linalg.solve(A, b)
    #Multiply the coefficient with the vandermonde matrix to find the result
    
    """
    if valid is None:
        valid=[[0,0],[0,0]]
    from matplotlib.pyplot import figure, imshow
    figure()
    imshow(valid)
    #"""
    
    return np.dot(np.moveaxis(vander,0,-1),c)
    
#def polyfit2d_alt(im, deg=[2,2], percentile=100):
#    """Fit the function f to the degree deg
#    
#    Parameters
#    ----------
#    im: 2d array
#        The image to fit
#    deg: 2 numbers, defaults [2,2]
#        The Y and X polynomial degrees to fit
#    percentile: number, optional
#        The percentage of the image covered by the background
#    
#    Returns
#    -------
#    im: 2d array
#        The fitted polynomial surface
#        
#    Notes
#    -----
#    Ignore everithing above percentile
#    This is kind of wrong as y^2 * x^2 is not 2nd degree...
#    """
#    #clean input
#    deg = np.asarray(deg)
#    im = np.asarray(im)  
#    im = cv2.GaussianBlur(im,(11,11),0)
#    #get x,y
#    x = np.asarray(range(im.shape[1]))
#    y = np.asarray(range(im.shape[0]))
#    X = np.array(np.meshgrid(x,y))
#    #save shape
#    initshape=im.shape
#    #get vander matrix
#    vander = polynomial.polyvander2d(X[0], X[1], deg)
#    #reshape for lstsq
#    vander = vander.reshape((-1,vander.shape[-1]))
#    im = im.reshape((vander.shape[0],))
#    #get valid values
#    valid=im<np.nanpercentile(im,percentile)
#    #Find coefficients
#    c = np.linalg.lstsq(vander[valid,:], im[valid])[0]
#    #Compute value
#    ret=np.dot(vander,c)
#    return ret.reshape(initshape)

    
    #    vandermonde=np.zeros(((deg[0]*2+1),(deg[1]*2+1),*(im.shape)),dtype='float64')
#    for yp in range(deg[0]*2+1):
#        for xp in range(deg[1]*2+1):
#            #There is no clear need to recompute that each time
#            np.dot((y**yp),(x**xp),out=vandermonde[yp,xp,:,:])
#          
#    vanmonvalid=np.dot(vandermonde.reshape((*(vandermonde.shape[:2]),-1)),
#                                              valid.reshape((-1,)))
#    res=vanmonvalid.sum(-1)
