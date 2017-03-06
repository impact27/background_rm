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
warnings.filterwarnings('ignore', 'Mean of empty slice',RuntimeWarning)
import scipy
import scipy.ndimage.measurements as msr
sobel = scipy.ndimage.filters.sobel

def remove_curve_background(im, bg, deg=2, *, 
                            xOrientate=False, method=None, infoDict=None,
                            rotateAngle=None, mask=None, percentile=None):
    """flatten the image by removing the curve and the background fluorescence. 
    
    Parameters
    ----------
    im: 2d array
        The image with background and data
    bg: 2d array
        The image with only background
    percentile: number 0-100, optional
        The percentage of the image covered by the background
        If None, the script uses morphological functions to find the proteins
    deg: 2 numbers, default [2,2]
        The polynomial fit Y and X degrees
    xOrientate: boolean, default False
        if True, will orientate the image along the x axis
    method: string, defaults 'none'
        Flattening improvement method. Ignored if mask is provided
            'detectChannel': Detects the channel using outChannelMask
            'gaussianBeam':  Detects the proteins using outGaussianBeamMask
            'twoPass':       Applies two pass to detect the proteins 
    infoDict: dictionnary
        If not None, will contain the infos on modification
    rotateAngle: number
        If using xOrientate or the 'detectChannel' or 'gaussianBeam' methods,
        will determine the angle of the rotated image instead.
        Use if the original image has weak features and is oriented along 
        the axis.
    mask: 2d array of bool
        Part of the image to use for flattening
    
    Returns
    -------
    im: 2d array
        The image with background removed
        
    Notes
    -----
    for detectChannel, The channel should be clearly visible and straight.
    TODO: this function takes almost 2 seconds for a single pair of 1000x1000
    matrices!!! 1.2 of which for folyfit2d! need to optimize that
    Consider the background is flat (remove obvious dust)
    """
    #Save im and bg as float32
    im=np.asarray(im,dtype='float32')
    bg=np.asarray(bg,dtype='float32')

    
    #create mask to detect the background
    maskim=None
    maskbg=None
    if mask is not None:
        maskim=mask
        maskbg=mask  
    elif percentile is None:
        maskim=backgroundMask(im)
    elif percentile is not 100:
        maskim=im< np.nanpercentile(im,percentile)
    if mask is None:
        maskbg=backgroundMask(bg, nstd=6)
      
    #Flatten the image and background
    imI=polyfit2d(im,deg, mask=maskim)
    im=im/imI
    bg=bg/polyfit2d(bg,deg,mask=maskbg)
    
    #if the image has any nans, replace by 1 (for fft)
    nanim=np.isnan(im)
    nanbg=np.isnan(bg)
    im[nanim]=1
    bg[nanbg]=1

    #Detect the image angle if needed
    angleOri=0
    if xOrientate or method == 'detectChannel' or method == 'gaussianBeam':    
        angleOri=ir.orientation_angle(im,rotateAngle=rotateAngle)
     
    #get angle scale and shift
    angle, scale, shift, __ = ir.register_images(im,bg)
    
    #Reset the previously removed nans
    im[nanim]=np.nan
    bg[nanbg]=np.nan
    
    #move background
    bg=ir.rotate_scale_shift(bg,angle,scale,shift, borderValue=np.nan)
    
    #resize if shape is not equal
    if im.shape is not bg.shape:
        im, bg = same_size(im,bg)
        
    #subtract background
    data=im-bg
    
    #Get mask to flatten data
    if mask is not None:
        pass
    elif method == 'detectChannel':
        mask= outChannelMask(bg,angleOri)   
    elif method == 'twoPass':
        mask=backgroundMask(data,im.shape[0]//100,blur=True)
    elif method == 'gaussianBeam':
        mask= outGaussianBeamMask(data,angleOri)
    
    #Flatten data
    if mask is not None:
       data+=1
       data/=polyfit2d(data,deg,mask=mask)
       data-=1
        
               
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
    if chAngle!=0:
        data=ir.rotate_scale(data, -chAngle,1,np.nan)
    profile=np.nanmean(data,1)
    
    #guess position of max
    amax= profile.size//2
    
    #get X and Y
    X0=np.arange(profile.size)-amax
    Y0=profile
    
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
    
    if chAngle!=0:
        idx=np.indices(mask.shape)
        
        
        idx[1]-=mask.shape[1]//2
        idx[0]-=mask.shape[0]//2
        X=np.cos(chAngle)*idx[1]+np.sin(chAngle)*idx[0]
        Y=np.cos(chAngle)*idx[0]-np.sin(chAngle)*idx[1]
        
        mask[np.abs(Y-mean+mask.shape[0]//2)<dist]=0
        
    else:    
        mask[llim:rlim,:]=0
    
    #mask=np.logical_and(mask>.5, np.isfinite(data))
    mask=mask>.5
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
#@profile
def polyfit2d(im,deg=2, *,x=None, y=None, mask=None):
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
    psize=((deg+1)*(deg+2))//2
    #vandermonde matrix
    vander=np.empty((psize,*(im.shape)),dtype='float32')
        
    c=polyfit2dcoeff(im, deg=deg, mask=mask, vanderOut=vander,x=x,y=y)
    
    #Multiply the coefficient with the vandermonde matrix to find the result
    ret=np.zeros(im.shape,dtype=c.dtype)
    for coeff, mat in zip(c,vander):
        ret+=coeff*mat
    return ret

def getidx(y,x, deg=2):
    """function to order powers in psize"""
    return ((2*(deg+1)+1)*y-y**2)//2+x
           
#@profile                
def polyfit2dcoeff(im, *,x=None, y=None, deg=2, mask=None, vanderOut=None):
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
    if x is None:
        x=range(im.shape[1])
    else:
        assert(len(x)==im.shape[1])
    if y is None:
        y=range(im.shape[0])
    else:
        assert(len(y)==im.shape[0])
    #get x,y
    x = np.asarray(x,dtype='float32')[np.newaxis,:]
    y = np.asarray(y,dtype='float32')[:,np.newaxis]
    
    valid=np.isfinite(im)
    if mask is not None:
        valid=np.logical_and(valid,mask)
        
    if np.all(valid):
        valid=None
        
    
    #Number of x and y power combinations
    psize=((deg+1)*(deg+2))//2
    #This will hold the sum of the vandermonde matrix, 
    #square instead of shape(psize) for readibility.
    #This will therefore be a upper triangular matrix
    SOPV=np.empty(((deg*2+1),(deg*2+1)),dtype='float32')
    if vanderOut is not None:
        assert(vanderOut.shape==(psize,*(im.shape)))
        vander=vanderOut
    else:
        #vandermonde matrix
        vander=np.empty((psize,*(im.shape)),dtype='float32')
    #vandermonde matrix with all masked values =0
    vandermasked=np.empty((psize,*(im.shape)),dtype='float32')
    #Temp. matrix that will hold the current value of vandermonde
    vtmp=np.empty(im.shape,dtype='float32')
    #idem but with 0 on masked pixels
    vtmpmasked=np.empty(im.shape,dtype='float32')

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
                vander[getidx(yp,xp,deg),:,:]=vtmp
                if valid is not None:
                    vandermasked[getidx(yp,xp,deg),:,:]=vtmpmasked
    
    #Then compute the LHS of the least square equation
    A=np.zeros((psize,psize),dtype='float32')
    for yi in range(deg+1):
        for yj in range(deg+1):
            for xi in range(deg+1-yi):
                for xj in range(deg+1-yj):
                    A[getidx(yi,xi,deg),getidx(yj,xj,deg)]=SOPV[yi+yj,xi+xj]
    
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
    
    """
    if valid is None:
        valid=[[0,0],[0,0]]
    from matplotlib.pyplot import figure, imshow
    figure()
    imshow(valid)
    #"""
    
    return c

def getPeaks(im, nstd=6, maxsize=None):
    """
    Detects the peaks using edge detection
    Parameters
    ----------
    im: 2d array
        The image to fit
    
    Returns
    -------
    peaks: 2d array
        mask of the peaks location
        
    Notes
    -----
    Position of high slope and intern parts are marked
    """
    im=np.asarray(im,dtype='float')
    imblur=np.empty(im.shape)
    edge=cr.Scharr_edge(im, imblur=imblur)
    threshold=np.nanmean(edge)+6*np.nanstd(edge)
    peaks=edge>threshold
    
    labels,n=msr.label(peaks)
    intensity_inclusions=msr.mean(imblur,labels,np.arange(n)+1)
    
    for i in np.arange(n)+1:
        if intensity_inclusions[i-1] > np.nanmean(imblur):
            high, m=msr.label(imblur>intensity_inclusions[i-1])
            for j in np.unique(high[np.logical_and(labels==i,high>0)]):
                labels[high==j]=i
            
            if maxsize is not None and np.sum(labels==i)>maxsize:
                labels[labels==i]=0
        else:
            labels[labels==i]=0
            
            
    """
    from matplotlib.pyplot import figure, hist, plot, title, ylim
    figure()
    hist(edge[np.isfinite(edge)],100)
    plot(threshold*np.array([1,1]),[0,ylim()[1]])
    title(str(threshold))
    #"""
    return labels>0     

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
