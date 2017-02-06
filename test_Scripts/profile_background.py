# -*- coding: utf-8 -*-
"""
Profiles remove_curve_background

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
import background_rm as rmbg
import matplotlib.image as mpimg
import importlib
importlib.reload(rmbg)

#%% load images
im=mpimg.imread('../Data/UVData/im0.tif')
bg=mpimg.imread('../Data/UVData/ba_e1105qt5_500ms.tif')
method='twoPass'
#%%
for i in range(1):
    rmbg.remove_curve_background(bg,im,xOrientate=True, method=method)
#    rmbg.polyfit2d(im,2)