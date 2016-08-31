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

#%% load images
fns=['Data/UVData/im0.tif']
fns.append('Data/UVData/ba_e1105qt5_500ms.tif')
fns.append('Data/UVData/ba_e1105qt5bg2_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]
    

#%%
for i in range(10):
    rmbg.remove_curve_background(imgs[1],imgs[0],xOrientate=True, twoPass=True)