# -*- coding: utf-8 -*-

import background_rm as rmbg
import matplotlib.image as mpimg

#%% load images
fns=['UVData/im0.tif']
fns.append('UVData/ba_e1105qt5_500ms.tif')
fns.append('UVData/ba_e1105qt5bg2_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]
    
#%%
for i in range(2):
    rmbg.polyfit2d(imgs[1],(2,2))
    rmbg.polyfit2d2(imgs[1],(2,2))
#%%
#for i in range(1):
#    rmbg.remove_curve_background(imgs[1],imgs[0],90)