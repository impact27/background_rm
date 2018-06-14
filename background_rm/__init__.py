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
import registrator.image as ir
import registrator.channel as cr
import numpy as np
import cv2
from scipy.special import erfinv
import warnings
import scipy
import scipy.ndimage.measurements as msr
sobel = scipy.ndimage.filters.sobel
warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
warnings.filterwarnings('ignore', 'invalid value encountered in less',
                        RuntimeWarning)


def remove_curve_background(im, bg, deg=2, *,
                            maskim=None, maskbg=None, infoDict=None,
                            bgCoord=False, percentile=None,
                            reflatten=True, use_bg_curve=False,
                            correctScale=False):
    """flatten the image by removing the curve and the background fluorescence.

    Parameters
    ----------
    im: 2d array
        The image with background and data
    bg: 2d array
        The image with only background
    deg: integer, default 2
        The polynomial degree to flatten the image
    maskim: 2d array of bool, default None
        Part of the image to use for flattening
    maskbg: 2d array of bool, default None
        Part of the background to use for flattening,
        if None, takes the value of maskim
    infoDict: dictionnary
        If not None, will contain the infos on modification
    bgCoord: bool, default False
        Use background image coordinates instead of image coordinates
    percentile: number 0-100, optional
        The percentage of the image covered by the background
        It is used if mskim is not given to try to ignore the signal
    reflatten: Bool, default True
        Reflatten after background subtraction to get a flatter result

    Returns
    -------
    output: 2d array
        The image with background removed
    """
    # Save im and bg as float32
    im = np.asarray(im, dtype='float32')
    bg = np.asarray(bg, dtype='float32')
    
    squeeze = False
    if len(np.shape(im)) == 2:
        squeeze = True
        im = im[np.newaxis]

    # Get the masks in order
    if maskim is not None and maskbg is None:
        maskbg = maskim
    elif maskbg is not None and maskim is None:
        maskim = maskbg

    # if no mask is provided, guess them
    twoPass = False
    if maskim is None:
        twoPass = True
        single_im = np.nanmean(im,0)
        if percentile is None:
            maskim = backgroundMask(single_im)
        elif percentile is not 100:
            maskim = single_im < np.nanpercentile(single_im, percentile)
        maskbg = backgroundMask(bg, nstd=6)

    # Flatten the image and background
    fim = polyfit2d(im, deg, mask=maskim)
    fbg = polyfit2d(bg, deg, mask=maskbg)

    if np.any(fim <= 0):
        raise RuntimeError("Image mask too small")

    if np.any(fbg <= 0):
        raise RuntimeError("Background mask too small")

    if use_bg_curve:
        factor = (np.mean(fbg[maskim])/np.mean(fim[..., maskim], -1))
        fim = fbg / factor[..., np.newaxis, np.newaxis]

    if infoDict is not None:
        infoDict['image_intensity'] = fim
    
    im = im / fim
    bg = bg / fbg
    
    data = align(im, bg, bgCoord, infoDict, correctScale=correctScale)

    fdata = 1
    # Get mask to flatten data
    if reflatten:
        if twoPass:
            single_data = np.nanmean(data, 0)
            mask = backgroundMask(single_data, im.shape[-2] // 100, blur=True)
        elif bgCoord:
            mask = maskbg
        else:
            mask = maskim
        # Flatten data
        data += 1
        fdata = polyfit2d(data, deg, mask=mask)
        data /= fdata
        data -= 1
        if infoDict is not None:
            infoDict['image_intensity_reflatten'] = fdata

    

    """
    from matplotlib.pyplot import figure, imshow
    figure()
    imshow(im)
    imshow(maskim,alpha=.5,cmap='Reds')
    figure()
    imshow(bg)
    imshow(maskbg,alpha=.5,cmap='Reds')
    #"""
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

    # return result
    if squeeze:
        data = np.squeeze(data)
        if infoDict is not None:
            infoDict['diffAngle'] = np.squeeze(infoDict['diffAngle'])
            infoDict['diffScale'] = np.squeeze(infoDict['diffScale'])
            infoDict['offset'] = np.squeeze(infoDict['offset'])
            infoDict['image_intensity'] = np.squeeze(
                    infoDict['image_intensity'])
            infoDict['offset'] = np.squeeze(infoDict['offset'])
            infoDict['image_intensity_reflatten'] = np.squeeze(
                    infoDict['image_intensity_reflatten'])
    return data

def align(images, background, bgCoord=False, infoDict=None, correctScale=True):
    # if the image has any nans (for fft)
    nanim = np.isnan(images)
    nanbg = np.isnan(background)
    
    data = np.empty_like(images)
    
    if infoDict is not None:
        infoDict['diffAngle'] = []
        infoDict['diffScale'] = []
        infoDict['offset'] = []
    
    for i, im in enumerate(images):
        bg = background.copy()
        # Replace nans to do registration
        im[nanim[i]] = 1
        bg[nanbg] = 1
        
        # get angle scale and shift
        angle, scale, shift, __ = ir.register_images(im, bg, 
                                                     correctScale=correctScale)
    
        # Reset the previously removed nans
        im[nanim[i]] = np.nan
        bg[nanbg] = np.nan
    
        if bgCoord:
            # move image
            im = ir.rotate_scale_shift(im, -angle, 1 / scale, -np.array(shift),
                                       borderValue=np.nan)
        else:
            # move background
            bg = ir.rotate_scale_shift(bg, angle, scale, shift,
                                       borderValue=np.nan)
        
        if infoDict is not None:
            infoDict['diffAngle'].append(angle)
            infoDict['diffScale'].append(scale)
            infoDict['offset'].append(shift)
            
            # resize if shape is not equal
        if im.shape != bg.shape:
            im, bg = same_size(im, bg)
    
        # subtract background
        data[i] = im - bg
        
    return data
    
def same_size(ims, bg):
    """Pad with nans to get similarely shaped matrix

    Parameters
    ----------
    ims: 2d array
        First image
    bg: 2d array
        Second image

    Returns
    -------
    ims: 2d array
        First image
    bg: 2d array
        Second image

    """
    shape = [max(ims.shape[-2], bg.shape[-2]), 
             max(ims.shape[-1], bg.shape[-1])]
    
    imshape = np.array(np.shape(ims))
    imshape[-2:] = shape
    bgshape = np.array(np.shape(bg))
    bgshape[-2:] = shape
    
    ims_out = np.empty(tuple(imshape)) * np.nan
    ims_out[..., :ims.shape[-2], :ims.shape[-1]] = ims
    
    bg_out = np.empty(tuple(bgshape)) * np.nan
    bg_out[..., :bg.shape[-2], :bg.shape[-1]] = bg
    
#    ims = cv2.copyMakeBorder(ims, 0, shape[0] - ims.shape[0],
#                             0, shape[1] - ims.shape[1],
#                             borderType=cv2.BORDER_CONSTANT, value=np.nan)
#    bg = cv2.copyMakeBorder(bg, 0, shape[0] - bg.shape[0],
#                             0, shape[1] - bg.shape[1],
#                             borderType=cv2.BORDER_CONSTANT, value=np.nan)

    return ims_out, bg_out


def backgroundTreshold(im, nstd=3):
    """get a threshold to remove background

    Parameters
    ----------
    im: 2d array
        The image
    nstd: integer
        Number of standard deviations to use as a threshold

    Returns
    -------
    threshold: number
        the threshold

    """
    # take mode

    immin = np.nanmin(im)
    immax = np.nanmax(im)
    hist, *_ = np.histogram(im[np.isfinite(im)], 1000, [immin, immax])
    m = hist[1:-1].argmax() + 1  # don't want saturated values
    hm = m
    m = m * (immax - immin) / 1000 + immin
    # if hist 0 is saturated, use erfinv
    if hist[0] > 0.5 * hist.max():
        std = (-m) / np.sqrt(2) / erfinv(hist[0] / hist[:hm].sum() - 1)
    else:
        # std everithing below mean
        std = np.sqrt(((m - im[im < m])**2).mean())

    return m + nstd * std


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
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def backgroundMask(im, r=2, blur=False, nstd=3):
    """Tries to extract the background of the image

    Parameters
    ----------
    im: 2d array
        The image
    r: uint
        The radius used to fill the gaps
    blur: Bool, default False
        Should the image be blurred before processing?
    nstd: integer
        Number of standard deviations to use as a threshold

    Returns
    -------
    valid: 2d array
        the valid mask

    """
    finite = np.isfinite(im)
    if blur:
        im = cv2.GaussianBlur(im, (2 * r + 1, 2 * r + 1), 0)
    valid = im < backgroundTreshold(im, nstd)
    valid = np.asarray(valid, dtype="uint8")

    # remove dots in proteins (3px dots)
    valid = cv2.erode(valid, getCircle(r))
    # remove dots in background (2 px dots)
    valid = cv2.dilate(valid, getCircle(r + r // 2))
    # widen proteins (10 px around proteins)
    valid = cv2.erode(valid, getCircle(r))

    # If invalid values in im, get rid of them
    valid = np.logical_and(valid, finite)

    return valid


def signalMask(im, r=2, blur=False):
    """Tries to extract the signal of the image

    Parameters
    ----------
    im: 2d array
        The image
    r: uint
        The radius used to fill the gaps
    blur: Bool, default False
        Should the image be blurred before processing?

    Returns
    -------
    valid: 2d array
        the valid mask

    """
    finite = np.isfinite(im)
    if blur:
        im = cv2.GaussianBlur(im, (2 * r + 1, 2 * r + 1), 0)

    valid = im > backgroundTreshold(im)
    valid = np.asarray(valid, dtype="uint8")

    # remove dots in proteins (3px dots)
    valid = cv2.dilate(valid, getCircle(r))
    # remove dots in background (2 px dots)
    valid = cv2.erode(valid, getCircle(r + r // 2))
    # widen proteins (10 px around proteins)
    valid = cv2.dilate(valid, getCircle(r // 2))

    # If invalid values in im, get rid of them
    valid = np.logical_and(valid, finite)

    return valid


#@profile
def polyfit2d(ims, deg=2, *, x=None, y=None, mask=None):
    """Fit the function f to the degree deg

    Parameters
    ----------
    im: 2d array
        The image to fit
    deg: integer, default 2
        The polynomial degree to fit
    x: 1d array
        X position
    y: 1d array
        Y positions
    mask: 2d boolean array, default None
        Alternative to percentile, valid values

    Returns
    -------
    im: 2d array
        The fitted polynomial surface

    """
    ims = np.asarray(ims, 'float32')

    valid = np.isfinite(ims)
    if len(ims.shape) == 3:
        valid = np.sum(valid, 0) == ims.shape[0]
    if mask is not None:
        valid = np.logical_and(valid, mask)
    
    squeeze = False
    if len(np.shape(ims)) == 2:
        squeeze = True
        ims = ims[np.newaxis]

    if not hasattr(polyfit2d, 'LHS') or np.any(polyfit2d.valid != valid):
        polyfit2d.LHS = polyfit2dLHS(
                ims.shape[1:], deg=deg, mask=valid, x=x, y=y)
        polyfit2d.valid = valid
        
    vander, vandermasked, A = polyfit2d.LHS
    ret_array = np.zeros_like(ims)
    
    for ret, im in zip(ret_array, ims):
        c = solvePolyFit2d(im, valid, vandermasked, A)
        # Multiply the coefficient with the vandermonde matrix
        for coeff, mat in zip(c, vander):
            ret += coeff * mat
    if squeeze:
        ret_array = np.squeeze(ret_array)
    return ret_array

def getidx(y, x, deg=2):
    """helper function to order powers in psize

     Parameters
    ----------
    y: integer
        y position
    x: integer
        x position
    deg: integer, default 2
        The power

    Returns
    -------
    idx: integer
        The correct index"""
    return ((2 * (deg + 1) + 1) * y - y**2) // 2 + x



#@profile
def polyfit2dLHS(shape, *, x=None, y=None, deg=2, mask=None,
                   dtype='float32'):
    """Fit the function f to the degree deg

    Parameters
    ----------
    im: 2d array
        The image to fit
    x: 1d array
        X position
    y: 1d array
        Y positions
    deg: integer, default 2
        The polynomial degree to fit
    mask: 2d boolean array
        Alternative to percentile, valid values
    vanderOut: 2d array
        Outpts vander matrix if requested

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
    masked = not np.all(mask)
    if x is None:
        x = range(shape[1])
    else:
        assert(len(x) == shape[1])
    if y is None:
        y = range(shape[0])
    else:
        assert(len(y) == shape[0])
    # get x,y
    x = np.asarray(x, dtype=dtype)[np.newaxis, :]
    y = np.asarray(y, dtype=dtype)[:, np.newaxis]

    # Number of x and y power combinations
    psize = ((deg + 1) * (deg + 2)) // 2
    # This will hold the sum of the vandermonde matrix,
    # square instead of shape(psize) for readibility.
    # This will therefore be a upper triangular matrix
    SOPV = np.empty(((deg * 2 + 1), (deg * 2 + 1)), dtype=dtype)
    # vandermonde matrix
    vander = np.empty((psize, *shape), dtype=dtype)
    # vandermonde matrix with all masked values =0
    vandermasked = np.empty((psize, *shape), dtype=dtype)
    # Temp. matrix that will hold the current value of vandermonde
    vtmp = np.empty(shape, dtype=dtype)
    # idem but with 0 on masked pixels
    vtmpmasked = np.empty(shape, dtype=dtype)

    # First thing is to compute the vandermonde matrix
    for yp in range(deg * 2 + 1):
        for xp in range(deg * 2 + 1 - yp):
            # There is no clear need to recompute that each time
            np.dot((y**yp), (x**xp), out=vtmp)
            if masked:
                np.multiply(vtmp, mask, out=vtmpmasked)
                SOPV[yp, xp] = vtmpmasked.sum()
            else:
                SOPV[yp, xp] = vtmp.sum()

            if yp < deg + 1 and xp < deg + 1 - yp:
                vander[getidx(yp, xp, deg), :, :] = vtmp
                if masked:
                    vandermasked[getidx(yp, xp, deg), :, :] = vtmpmasked

    # Then compute the LHS of the least square equation
    A = np.zeros((psize, psize), dtype=dtype)
    for yi in range(deg + 1):
        for yj in range(deg + 1):
            for xi in range(deg + 1 - yi):
                for xj in range(deg + 1 - yj):
                    A[getidx(yi, xi, deg), getidx(yj, xj, deg)
                      ] = SOPV[yi + yj, xi + xj]
    
    if not masked:
        vandermasked = vander
             
    return vander, vandermasked, A
#@profile
def solvePolyFit2d(im, mask, vandermasked, A):

    # Set everithing invalid to 0 (as x*0 =0 works for any x)
    if not np.all(mask):
        im = im.copy()
        im[np.logical_not(mask)] = 0

    # Get the RHS of the least square equation
    b = np.dot(np.reshape(vandermasked, (vandermasked.shape[0], -1)),
               np.reshape(im, (-1,)))

    # Solve
    c = np.linalg.solve(A, b)

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
    nstd: integer, default 6
        Number of STD to use for theshold
    maxsize: integer
        Maximim size of the peak

    Returns
    -------
    peaks: 2d array
        mask of the peaks location

    Notes
    -----
    Position of high slope and intern parts are marked
    """
    im = np.asarray(im, dtype='float')
    imblur = np.empty(im.shape)
    edge = cr.Scharr_edge(im, imblur=imblur)
    threshold = np.nanmean(edge) + 6 * np.nanstd(edge)
    peaks = edge > threshold

    labels, n = msr.label(peaks)
    intensity_inclusions = msr.mean(imblur, labels, np.arange(n) + 1)

    for i in np.arange(n) + 1:
        if intensity_inclusions[i - 1] > np.nanmean(imblur):
            high, m = msr.label(imblur > intensity_inclusions[i - 1])
            for j in np.unique(high[np.logical_and(labels == i, high > 0)]):
                labels[high == j] = i

            if maxsize is not None and np.sum(labels == i) > maxsize:
                labels[labels == i] = 0
        else:
            labels[labels == i] = 0

    """
    from matplotlib.pyplot import figure, hist, plot, title, ylim
    figure()
    hist(edge[np.isfinite(edge)],100)
    plot(threshold*np.array([1,1]),[0,ylim()[1]])
    title(str(threshold))
    #"""
    return labels > 0
