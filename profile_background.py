# -*- coding: utf-8 -*-

import background_rm as rmbg
import matplotlib.image as mpimg

#%% load images
fns=['UVData/im0.tif']
fns.append('UVData/ba_e1105qt5_500ms.tif')
fns.append('UVData/ba_e1105qt5bg2_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]

#%%
for i in range(10):
    rmbg.remove_curve_background(imgs[1],imgs[0],90)