# -*- coding: utf-8 -*-
"""
test with other set of images

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
from scipy.special import erfinv
#%% Reload if changed

importlib.reload(rmbg)

#%% load images
im=mpimg.imread('Yuewen/25uM_Transferrin_resistance.tif')
bg=mpimg.imread('Yuewen/bg_resistance.tif')


#%%
d=rmbg.remove_curve_background(im,bg)
d2p=rmbg.remove_curve_background(im,bg,twoPass=True)
d3p=rmbg.remove_curve_background(im,bg,twoPass=True,threePass=True)
d100=d=rmbg.remove_curve_background(im,bg,percentile=100)

figure()
plot(np.nanmean(d,1), label = 'None')
plot(np.nanmean(d2p,1), label = '2pass')
plot(np.nanmean(d3p,1), label = '3pass')
plot(np.nanmean(d100,1), label = '100')
plot(np.zeros(np.nanmean(d,1).shape))
plt.legend()



#%%
fns=['UVData/im0.tif']
fns.append('UVData/ba_e1105qt5_500ms.tif')
fns.append('UVData/ba_e1105qt5bg2_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]
bg=imgs[1]
im=imgs[0]
#%%
f=rmbg.remove_curve_background(im,bg)
f0=f

#%%
figure()
imshow(f)
imshow(rmbg.getValid(cv2.GaussianBlur(f,(21,21),0)),alpha=.7)



#%%
figure()
plot(np.nanmean(im,1))
plot(np.nanmean(rmbg.polyfit2d(im),1))

#%%
figure()
plt.hist(im.ravel(),100);
#%%
imshow(im)

figure()
imshow(bg)

#%%
out=rmbg.remove_curve_background(im,bg,xOrientate=True)
figure()
imshow(out)

#%%
figure()
plot(np.nanmean(out,1))

#%%
out=rmbg.remove_curve_background(im,bg,xOrientate=True,twoPass=True)
figure()
imshow(out)

#%%
figure()
plot(np.nanmean(out,1))
#%%
f21=cv2.GaussianBlur(f,(21,21),0)
figure()
imshow(ret)
figure()
imshow(f21)
figure()
plt.hist(ret[np.isfinite(ret)],100)
figure()
plt.hist(f21[np.isfinite(f21)],100)
#%%
figure()
plt.hist(im.ravel(),100);
plt.plot([276,276],[0,40000])
plt.plot([358,358],[0,40000])
#%%
#take mode
immin=np.nanmin(im)
immax=np.nanmax(im)

hist,*_= np.histogram(im[np.isfinite(im)],1000,[immin,immax])
figure()
plt.plot(hist)
#%%
rmbg.polyfit2d(im)
#%%

f=cv2.GaussianBlur(f0,(51,51),0)
figure()
plt.hist(f[np.isfinite(f)],100)
#f=ret
m=np.nanmean(f)
fmin=np.nanmin(f)
fmax=np.nanmax(f)
#hist = cv2.calcHist([f[np.isfinite(f)]],[0],None,[1000],[fmin,fmax])
hist,*_= np.histogram(f[np.isfinite(f)],1000,[fmin,fmax])
m=hist[1:-1].argmax()+1

hm=m
m=m*(fmax-fmin)/1000+fmin
#if hist 0 is saturated, use erfinv
if hist[0]>0.5*hist.max():
    std=(-m)/np.sqrt(2)/erfinv(hist[0]/hist[:hm].sum()-1)
else:
    #std everithing below mean
    std=np.sqrt(((m-f[f<m])**2).mean())

plot([m+3*std,m+3*std],[0,30000])
valid=f<m+3*std

figure()
imshow(valid)

r=2
def turad(r):
    return (2*r+1,2*r+1)
#remove dots in proteins (3px dots)
valid=cv2.erode(np.asarray(valid,dtype="uint8"),np.ones(turad(r)))
#remove dots in background (2 px dots)
valid=cv2.dilate(valid,np.ones(turad(r+1)))
#widen proteins (10 px around proteins)
valid=cv2.erode(valid,np.ones(turad(r+1)))

figure()
imshow(f0)
imshow(valid,alpha=0.7)





#%%
figure()
plt.hist(f[np.isfinite(f)],100)
#%%
figure()
#imshow(f)
imshow(out2<0.1,alpha=0.5)
#%%
out2=cv2.GaussianBlur(out,(31,31),0)
figure()
plt.hist(out2[np.isfinite(out2)],100)
#%%
hist = cv2.calcHist([out2[np.isfinite(out2)]],[0],None,[200],[-1,1])
figure()
plot(hist)
m=hist.argmax()/100-1
s=np.sqrt(((m-out2[out2<m])**2).mean())
print(m,s,3*s)

#%%
figure()
imshow(out2)
#%%
figure()
imshow(out2<m+3*s,alpha=0.5)