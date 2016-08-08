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


#%% load images
fns=['UVData/im0.tif']
fns.append('UVData/ba_e1105qt5_500ms.tif')
fns.append('UVData/ba_e1105qt5bg2_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]

#%% Remove background
bg=imgs[1]
im=imgs[0]

im=ir.rotate_scale(im,np.pi/12,0.8)
data=rmbg.remove_curve_background(im,bg)

#%%
odata=rmbg.remove_curve_background(im,bg,xOrientate=True)


figure(0)
imshow(im)
#plt.imsave("im.png",im)

iim=rmbg.polyfit2d(im)
figure(1)
imshow(iim,vmin=im.min(),vmax=im.max())
#plt.imsave("iim.png",iim,vmin=im.min(),vmax=im.max())

figure(2)
imshow(bg)
#plt.imsave("bg.png",bg)

ibg=rmbg.polyfit2d(bg,percentile=100)
figure(3)
imshow(ibg,vmin=bg.min(),vmax=bg.max())
#plt.imsave("ibg.png",ibg,vmin=bg.min(),vmax=bg.max())

im=im/iim
bg=bg/ibg

angle, scale, shift, __ = ir.register_images(im,bg)

#move background

bg=zoom(bg,1/scale)

bg=rotate(bg,-angle*180/np.pi,cval=np.nan)

#%%
figure(4)
imshow(bg)
imshow(im,extent=ir.get_extent(-np.asarray(shift)+(np.asarray(bg.shape)-np.asarray(im.shape))/2, im.shape))
plt.autoscale()
#plt.savefig("superposed.png")

#%%

figure(5)
imshow(data,vmin=0)
#plt.imsave("result.png",data,vmin=0)
figure(6)
imshow(odata,vmin=0)
#plt.imsave("oresult.png",odata,vmin=0)

