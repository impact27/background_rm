# -*- coding: utf-8 -*-
"""
Creates the figures for the paper

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
from scipy.ndimage.interpolation import rotate,zoom
#%%
importlib.reload(ir)
importlib.reload(rmbg)

#%
bg=mpimg.imread('Data/Maya/12_background.tif')
im=mpimg.imread('Data/Maya/12.tif')
im=ir.rotate_scale(im,np.pi/12,.8,np.nan)
#%

data=rmbg.remove_curve_background(im,bg)

#%%
info={}
odata=rmbg.remove_curve_background(im,bg,xOrientate=True, twoPass=True, 
                                   infoDict=info)


#%%
figure(0)
imshow(im/np.nanmean(im)-1,vmin=-0.5,vmax=1)
#plt.imsave("im.png",cv2.resize(im/im.mean()-1,(500,500),interpolation=cv2.INTER_AREA),vmin=-0.5,vmax=1)

iim=rmbg.polyfit2d(im/np.nanmean(im))
figure(1)
imshow(iim-1,vmin=-0.5,vmax=1)
#plt.imsave("iim.png",cv2.resize(iim-1,(500,500),interpolation=cv2.INTER_AREA),vmin=-0.5,vmax=1)

figure(2)
imshow(bg/np.nanmean(bg)-1,vmin=-0.5,vmax=1)
#plt.imsave("bg.png",cv2.resize(bg/bg.mean()-1,(500,500),interpolation=cv2.INTER_AREA),vmin=-0.5,vmax=1)

ibg=rmbg.polyfit2d(bg/np.nanmean(bg))
figure(3)
imshow(ibg-1,vmin=-0.5,vmax=1)
#plt.imsave("ibg.png",cv2.resize(ibg-1,(500,500),interpolation=cv2.INTER_AREA),vmin=-0.5,vmax=1)

#%%

im2=im/iim/np.nanmean(im)
bg=bg/ibg/np.nanmean(bg)


#move background

bg=zoom(bg,1/info['diffScale'])

bg=rotate(bg,-info['diffAngle']*180/np.pi,cval=np.nan)

#%%
figure(4)
imshow(bg-1,vmin=-0.5,vmax=1)
im2[:5,:]=im2[-5:,:]=im2[:,:5]=im2[:,-5:]=0
imshow(im2-1,extent=ir.get_extent(-np.asarray(info['offset'])+(np.asarray(bg.shape)-np.asarray(im.shape))/2, im.shape),vmin=-0.5,vmax=1)
plt.autoscale()
plt.axis('off')
#plt.savefig("superposed.png")

#%%
datas=cv2.resize(data,(500,500),interpolation=cv2.INTER_AREA)
figure(5)
imshow(datas,vmin=-0.5,vmax=1)
#plt.imsave("result.png",datas,vmin=-0.5,vmax=1)
#%%

datac=cv2.copyMakeBorder(data,110,110,110,110,cv2.BORDER_CONSTANT,value=np.nan)
datac=ir.rotate_scale(datac,-info['imageAngle'],1,np.nan)
datacs=cv2.resize(datac,(600,600),interpolation=cv2.INTER_AREA)
figure(6)
imshow(datacs,vmin=-0.5,vmax=1)
#plt.imsave("oresult.png",datacs,vmin=-0.5,vmax=1)
#%%
close(7)

figure(7)

oim=ir.rotate_scale(im,-info['imageAngle'],1, borderValue=np.nan)/np.nanmean(im)
X=0.8*np.arange(oim.shape[0])
plot(X,np.nanmean(oim,1)-.8,label="Original image")
plot(X,np.nanmean(odata,1),label="Processed image")
plot([0,X[-1]],[0,0],'r--')
plot([0,X[-1]],[0.2,0.2],'r--')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Position [$\mu m$]')
plt.ylabel('Amplitude [unitless]')
#plt.savefig("compared.pdf")

#%%
oim2=cv2.resize(oim[300:700,:]-1,(500,200),interpolation=cv2.INTER_AREA)
odata2=cv2.resize(odata[300:700,:],(500,200),interpolation=cv2.INTER_AREA)
figure()
imshow(oim2,vmin=-.5,vmax=1)
#plt.imsave("oimzoom.png",oim2,vmin=-.5,vmax=1)
figure()
imshow(odata2,vmin=-.5,vmax=1)
#plt.imsave("odatazoom.png",odata2,vmin=-.5,vmax=1)
