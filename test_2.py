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
rmbg.polyfit2d(bg)
#%%
d=rmbg.remove_curve_background(im,bg)
d2p=rmbg.remove_curve_background(im,bg,twoPass=True)
d100=d=rmbg.remove_curve_background(im,bg,percentile=100)

figure()
plot(np.nanmean(d,1), label = 'None')
plot(np.nanmean(d2p,1), label = '2pass')
plot(np.nanmean(d100,1), label = '100')
plot(np.zeros(np.nanmean(d,1).shape))
plt.legend()


