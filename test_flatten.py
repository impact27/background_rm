# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:36:06 2016

@author: quentinpeter
"""

# -*- coding: utf-8 -*-

#%%
import sys
sys.path.append('../image_registration')
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy, hold
import matplotlib.image as mpimg
import numpy as np
import registration.image as ir
import registration.channel as cr
import background as bgrm
import importlib
import cv2
#%%
importlib.reload(ir)
importlib.reload(cr)
importlib.reload(bgrm)

#%% nload images
fns=['UVData/im0.tif']
fns.append('UVData/ba_e1105qt5_500ms.tif')
fns.append('UVData/ba_e1105qt5bg2_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]
    

#%%
im=imgs[0]
bg=imgs[1]


#%%
    


sub=bgrm.match_substract(im,bg)
div=bgrm.flatten_match(im,bg)


sub=sub[150:,30:-1]
div=div[150:,30:-1]

sub=(sub-bgrm.polyfit2d(sub,[2,2]))/im.mean()

figure()
plot(np.nanmean(sub,1))
plot(np.nanmean(div-1,1))
figure()
plot(np.nanmean(sub,0))
plot(np.nanmean(div-1,0))


#%
bluredim=cv2.GaussianBlur(im,(11,11),5)
figure()
imshow(bluredim>np.percentile(bluredim,95))
#"""
