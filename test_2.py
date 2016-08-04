# -*- coding: utf-8 -*-

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
#%% Reload if changed

importlib.reload(rmbg)

#%% load images
im=mpimg.imread('Yuewen/25uM_Transferrin_resistance.tif')
bg=mpimg.imread('Yuewen/bg_resistance.tif')

#%%
imshow(im)

figure()
imshow(bg)

#%%
out=rmbg.remove_curve_background(im,bg,xOrientate=True)
figure()
imshow(out)

#%%
figure()
plot(np.nanmean(out,1))

#%%
figure()
plot(np.nanmean(im,1))
plot(np.nanmean(rmbg.polyfit2d2(im),1))

#%%
f=im
m=f.mean()
s=np.sqrt(((m-f[f<m])**2).mean())

valid=f<m+3*s
#remove dots in proteins (3px dots)
valid=cv2.erode(np.asarray(valid,dtype="uint8"),np.ones((7,7)))
#remove dots in background (2 px dots)
valid=cv2.dilate(valid,np.ones((11,11)))
#widen proteins (10 px around proteins)
valid=cv2.erode(valid,np.ones((21,21)))
#If invalid values in f, get rid of them
valid=np.logical_and(valid,np.isfinite(f))

figure()
imshow(f)
imshow(valid,alpha=0.5)




#%%
figure()
plt.hist(f[np.isfinite(f)],100)
#%%
