# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:33:28 2016

@author: quentinpeter
"""

#%% Imports
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy, hold,colorbar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import image_registration.image as ir
import image_registration.channel as cr
import background_rm as rmbg
import importlib
import cv2
import warnings
import scipy
gfilter=scipy.ndimage.filters.gaussian_filter1d

#%% Reload if changed
importlib.reload(ir)
importlib.reload(cr)
importlib.reload(rmbg)

#%% Load Images
imfn='Data/Maya_Images/im_1.tif'
bgfn='Data/Maya_background/1.tif'


im=mpimg.imread(imfn)
bg=mpimg.imread(bgfn)

info={}
data=rmbg.remove_curve_background(im,bg,infoDict=info, detectChannel=True,xOrientate=True)

bg=ir.rotate_scale_shift(bg,info['diffAngle'],info['diffScale'],info['offset'], np.nan)

#%%
im0=im.copy()
bg0=bg.copy()

figure()
plot(np.nanmean(data,1))
plot([0,data.shape[1]],[0,0],'r')

#create mask to detect the background
mask=rmbg.backgroundMask(im0)
  
#Flatten the imaget   
im0=im0/rmbg.polyfit2d(im0, mask=mask)#

#if the image has any nans, replace by 1 (for fft)
nanim=np.isnan(im0)
im0[nanim]=1

#Detect the image angle if needed
angleOri=ir.orientation_angle(im0)

#If detect channel, correct with channel and proceed
mask= rmbg.outChannelMask(bg0,angleOri)

im0=im0/rmbg.polyfit2d(im0,mask=mask)
bg0=bg0/rmbg.polyfit2d(bg0,mask=mask)

#Correct the background for nan too
nanbg=np.isnan(bg0)
bg0[nanbg]=1
 
#get angle scale and shift
angle, scale, shift, __ = ir.register_images(im0,bg0)

#remove the previously added nans
im0[nanim]=np.nan
bg0[nanbg]=np.nan

#move background
bg0=ir.rotate_scale_shift(bg0,angle,scale,shift, borderValue=np.nan)
bg2=ir.rotate_scale_shift(bg,angle,scale,shift, borderValue=np.nan)
#resize if shape is not equal
if im0.shape is not bg0.shape:
    im0, bg0 = rmbg.same_size(im0,bg0)
    
#subtract background
data=im0-bg0

#If 2 pass, get flatter image and resubtract    

           
#if we want to orientate the image, do it now
data=ir.rotate_scale(data,-angleOri,1, borderValue=np.nan)
  
#%
#"""
from matplotlib.pyplot import figure, plot

im2=ir.rotate_scale(im,-angleOri,1, borderValue=np.nan)
bg2=ir.rotate_scale(bg2,-angleOri,1, borderValue=np.nan)
im0=ir.rotate_scale(im0,-angleOri,1, borderValue=np.nan)
bg0=ir.rotate_scale(bg0,-angleOri,1, borderValue=np.nan)
mask2=ir.rotate_scale(mask,-angleOri,1, borderValue=np.nan)>.5

#%%
figure()
plot(np.nanmean(im2,1),'b')
plot(np.nanmean(bg2,1),'g')
plot(np.nanmean(rmbg.polyfit2d(im2,mask=mask2),1),'b')
plot(np.nanmean(rmbg.polyfit2d(bg2,mask=mask2),1),'g')

figure()
plot(np.nanmean(im0,1),'b')
plot(np.nanmean(bg0,1),'g')
plot(np.nanmean(data,1)+1,'r')
plot([0,im0.shape[0]],[1,1])

#%%
figure()
imshow(im)
figure()
imshow(bg)
figure()
imshow(data)

