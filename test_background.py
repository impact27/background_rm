# -*- coding: utf-8 -*-

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
#%% Reload if changed
importlib.reload(ir)
importlib.reload(cr)
importlib.reload(rmbg)

#%% load images
fns=['UVData/im0.tif']
fns.append('UVData/ba_e1105qt5_500ms.tif')
fns.append('UVData/ba_e1105qt5bg2_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]

#%% Plot images

for im in imgs:
    figure()
    imshow(im)

#%% Remove background
bg=imgs[1]
im0=imgs[2]
im1=imgs[0]

#remove noise from background to reduce noise
#bg=cv2.GaussianBlur(bg,(3,3),1)

data0=rmbg.remove_curve_background(im0,bg,percentile=90)
data1=rmbg.remove_curve_background(im1,bg,percentile=90)

#%%
figure()
imshow(rmbg.polyfit2d(im1,[2,2],90))
figure()
imshow(rmbg.polyfit2d2(im1,[2,2],90))
#%% Plot images with background removed
# im0
figure()
p=imshow(data0,vmin=0,vmax=1)
colorbar(p)
#im 1
figure()
p=imshow(data1,vmin=0,vmax=1)
colorbar(p)
#im 1 with gaussian filter to remove noise
figure()
p=imshow(cv2.GaussianBlur(data1,(3,3),1),vmin=0,vmax=1)
colorbar(p)
#im1 without processing
figure()
p=imshow(im1)
colorbar(p)

#%%Compare profile
for data, im in zip((data0,data1),(im0,im1)):
    i=np.nanargmax(np.nanmean(data,1))
    figure()
    plot(im[i,:]/im.mean(),label = "image +1")
    plot(data[i,:], label= "extracted")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

i1=np.nanargmax(np.nanmean(data1,1))
figure()
plot(cv2.GaussianBlur(im1/im1.mean(),(3,3),1)[i1,:],label = "blured image +1")
plot(cv2.GaussianBlur(data1,(3,3),1)[i1,:], label= "blured extracted")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

#%% Compare x-mean two images
figure()
p0=np.nanmean(data0,1)
plot(p0[np.isfinite(p0)],label='image 0')
p1=np.nanmean(data1,1)
plot(p1[np.isfinite(p1)], label = 'image 1')
plt.legend()

#%% Test the usefulness of the percentile
data1noperc=rmbg.remove_curve_background(im1,bg,percentile=100)

figure()
plot(np.nanmean(data1,1), label = 'With percentile')
plot(np.nanmean(data1noperc,1), label = 'Without percentile')
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