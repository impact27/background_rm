# -*- coding: utf-8 -*-
"""
Test with some images 

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
#%% Reload if changed
importlib.reload(ir)
importlib.reload(cr)
importlib.reload(rmbg)

#%% load images
fns=['../Data/UVData/im0.tif']
fns.append('../Data/UVData/ba_e1105qt5_500ms.tif')
fns.append('../Data/UVData/ba_e1105qt5bg2_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]

#%%
im=imgs[0]
figure()
imshow(rmbg.polyfit2d(im))
colorbar()
figure()
imshow(rmbg.polyfit2d(cv2.copyMakeBorder(im,5,5,5,5,cv2.BORDER_CONSTANT,value=np.nan)))
colorbar()
#%% Plot images
#"""
for im in imgs:
    figure()
    imshow(im)
    colorbar()
#"""

#%% Remove background
bg=imgs[1]
im0=imgs[2]
im1=imgs[0]

#remove noise from background to reduce noise
#bg=cv2.GaussianBlur(bg,(3,3),1)

data0=rmbg.remove_curve_background(im0,bg)
data1=rmbg.remove_curve_background(im1,bg)

#%% Plot images with background removed
# im0
figure()
p=imshow(data0,vmin=0,vmax=1)
colorbar(p)
figure()
p=imshow(cv2.GaussianBlur(data0,(3,3),1),vmin=0,vmax=1)
colorbar(p)

#%%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m=np.nanargmax(np.nanmean(data1,1))
d=100

#im 1
figure()
p=imshow(data1[m-d:m+d,:],vmin=0,vmax=1)
#plt.imsave("Processed.png",data1[m-d:m+d,:],dpi=100,vmin=0,vmax=1)
#colorbar(p,orientation="horizontal")
#im 1 with gaussian filter to remove noise
figure()
p=imshow(cv2.GaussianBlur(data1,(3,3),1)[m-d:m+d,:],vmin=0,vmax=1)
#plt.imsave("ProcessedBlur.png",cv2.GaussianBlur(data1,(3,3),1)[m-d:m+d,:],dpi=100,vmin=0,vmax=1)
#colorbar(p)
#im1 without processing
figure()
p=imshow((im1/im1.mean()-1)[m-d:m+d,:],vmin=0,vmax=1)
#plt.imsave("Original.png",(im1/im1.mean()-1)[m-d:m+d,:],dpi=100,vmin=0,vmax=1)
#colorbar(p)

#im1 without processing
figure()
p=imshow(cv2.GaussianBlur(im1/im1.mean()-1,(3,3),1)[m-d:m+d,:],vmin=0,vmax=1)
#plt.imsave("OriginalBlur.png",cv2.GaussianBlur(im1/im1.mean()-1,(3,3),1)[m-d:m+d,:],dpi=100,vmin=0,vmax=1)
#colorbar(p)

#%%
vmin=-.3
vmax=1

image=im1/im1.mean()-1

figure()
imshow(image,vmin=vmin,vmax=vmax)
#plt.imsave("OriginalPoly.png",image,vmin=vmin,vmax=vmax)


image=rmbg.polyfit2d(image)

figure()
imshow(image,vmin=vmin,vmax=vmax)
#plt.imsave("PolyFit.png",image,vmin=vmin,vmax=vmax)

#%%Compare profile
for data, im in zip((data0,data1),(im0,im1)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i=np.nanargmax(np.nanmean(data,1))
    figure()
    plot(im[i,:]/im.mean(),label = "image +1")
    plot(data[i,:], label= "extracted")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    i1=np.nanargmax(np.nanmean(data1,1))
figure()
plot(cv2.GaussianBlur(im1/im1.mean(),(3,3),1)[i1,:],label = "blured image +1")
plot(cv2.GaussianBlur(data1,(3,3),1)[i1,:], label= "blured extracted")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

#%%
figure()
d=10
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m=np.nanmean(data1[i1-d:i1+d,:])
plot(cv2.GaussianBlur(data1,(3,3),1)[i1-d:i1+d,:1000].mean(0),'b', label= "Processed")
plot(np.ones((1000,))*m,'r--')
plot((im1/im1.mean())[i1-d:i1+d,:1000].mean(0)-0.5,'g',label = "Raw +.5")
plot(np.ones((1000,))*m+0.5,'r--')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.xlabel('position')
plt.ylabel('intensity')

#%% Compare x-mean two images
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    figure()
    p0=np.nanmean(data0,1)
    plot(p0[np.isfinite(p0)],label='image 0')
    p1=np.nanmean(data1,1)
    plot(p1[np.isfinite(p1)], label = 'image 1')
    plt.legend()

#%% Test the usefulness of the percentile
data1=rmbg.remove_curve_background(im1,bg)
data1100=rmbg.remove_curve_background(im1,bg,percentile=100)
data195=rmbg.remove_curve_background(im1,bg,percentile=95)
data12p=rmbg.remove_curve_background(im1,bg,method='twoPass')
dataGB=rmbg.remove_curve_background(im1,bg,method='gaussianBeam')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    figure()
    plot(np.nanmean(data1,1)+.1,'b', label = 'With percentile')
    plot(np.zeros(np.nanmean(data1,1).shape)+.1,'b')
    plot(np.nanmean(data1100,1)+.2,'r', label = 'Without percentile')
    plot(np.zeros(np.nanmean(data1,1).shape)+.2,'r')
    plot(np.nanmean(data195,1)+.3,'g', label = '95')
    plot(np.zeros(np.nanmean(data1,1).shape)+.3,'g')
    plot(np.nanmean(data12p,1)+.4,'k', label = '2pass')
    plot(np.zeros(np.nanmean(data1,1).shape)+.4,'k')
    plot(np.nanmean(dataGB,1)+.5,'y', label = 'GaussianBeam')
    plot(np.zeros(np.nanmean(data1,1).shape)+.5,'y')
    plt.legend()

#%% Rough Comparison of Processed and unprocessed images
figure()
plot(im1.mean(1)[np.isfinite(p1)]/im1.mean()-1, label= 'Unprocessed')
plot(p1[np.isfinite(p1)], label = 'Processed')
plt.legend()
#%% TEst of not straight image
im=ir.rotate_scale(im1,np.pi/20,1, borderValue=np.nan)
figure()
imshow(im)

#
data=rmbg.remove_curve_background(im,bg,percentile=90,xOrientate=True)
figure()
imshow(data,vmin=0,vmax=1)

##%%
#fit0=rmbg.polyfit2d(im0,[2,2],90)
#fit1=rmbg.polyfit2d2(im0,[2,2],90)
#figure()
#p=imshow(fit0)
#colorbar(p)
#figure()
#p=imshow(fit1)
#colorbar(p)
#
#figure()
#plot(im0.mean(1),label='')
#plot(fit0.mean(1))
#figure()
#plot(im0.mean(1))
#plot(fit1.mean(1))
