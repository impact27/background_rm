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

#
##%% load images
#fns=['UVData/im0.tif']
#fns.append('UVData/ba_e1105qt5_500ms.tif')
#fns.append('UVData/ba_e1105qt5bg2_1000ms.tif')
#imgs=[mpimg.imread(fn) for fn in fns]
#
##%% Remove background
#bg=imgs[1]
#im=imgs[0]

#%%
bg=mpimg.imread('Data/Maya/3_background.tif')
im=mpimg.imread('Data/Maya/3.tif')


#get angle scale and shift
angle, scale, shift, __ = ir.register_images(im,bg)
bg=ir.rotate_scale_shift(bg,angle,scale,shift, borderValue=np.nan)
#subtract background
data=im-bg
angleOri=ir.orientation_angle(im)
data=ir.rotate_scale(data,-angleOri,1, borderValue=np.nan)

figure()
plot(np.nanmean(data,1))

figure()
plot(np.nanmean(im,0))
plot(np.nanmean(bg,0))
