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
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy
from matplotlib.pyplot import hold, hist, colorbar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import image_registration.image as ir
import image_registration.channel as cr
import background_rm as rmbg
import importlib
import cv2
from scipy.special import erfinv
from skimage.filters import threshold_otsu
import warnings

def printInfo(d,m):
    print('mean',d[m].mean(),
          'std',d[m].std(),
          'std/SQRT(N)',d[m].std()/np.sqrt(m.sum()))
#%% Reload if changed

importlib.reload(rmbg)

#%% load images
im=mpimg.imread('Data/Yuewen/25uM_Transferrin_resistance.tif')
bg=mpimg.imread('Data/Yuewen/bg_resistance.tif')

#%%
im=mpimg.imread('Data/Yuewen/12-1.tif')
bg=mpimg.imread('Data/Yuewen/12.tif')

#%%
rmbg.polyfit2d(bg)
#%%
d=rmbg.remove_curve_background(im,bg,xOrientate=True)
d2p=rmbg.remove_curve_background(im,bg,twoPass=True,xOrientate=True)
d100=d=rmbg.remove_curve_background(im,bg,percentile=100,xOrientate=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    figure()
    plot(np.nanmean(d,1), label = 'None')
    plot(np.nanmean(d2p,1), label = '2pass')
    plot(np.nanmean(d100,1), label = '100')
    plot(np.zeros(np.nanmean(d,1).shape))
    plt.legend()

#%%
figure()
imshow(d2p)
#%%
sm=rmbg.signalMask(d2p,10,True)
figure()
imshow(sm)
printInfo(d2p,sm)


#%%
bm=rmbg.backgroundMask(d2p,10,True)
figure()
imshow(bm)
printInfo(d2p,bm)

#%%
a=im/rmbg.polyfit2d(im,mask=rmbg.backgroundMask(im))
b=bg/rmbg.polyfit2d(bg)
figure()
imshow(a)

figure()
imshow(b)

#%%

figure()
plot(np.mean(a,0))
plot(np.mean(b,0))
plot(np.nanmean(d2p,1)[::-1]+1)
#%%
d2p=rmbg.remove_curve_background(im,bg,twoPass=True,xOrientate=True)
