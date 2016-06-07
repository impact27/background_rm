# -*- coding: utf-8 -*-

#%%
import sys
sys.path.append('../chreg')
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy, hold
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import registration.image as ir
import registration.channel as cr
from background import match_substract
import importlib
#%%
importlib.reload(ir)
importlib.reload(cr)

#%% nload images
fns=['UVData/im0.tif']
fns.append('UVData/ba_e1105qt5_500ms.tif')
fns.append('UVData/ba_e1105qt5bg2_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]

#%%

for im in imgs:
    figure()
    imshow(im)
    
#%%

#%%
bg=imgs[1]
im0=imgs[2]
im1=imgs[0]
data0=match_substract(im0,bg)
figure()
imshow(data0)
data1=match_substract(im1,bg)
figure()
imshow(data1)
#%%
figure()
p0=np.nanmean(data0,1)
plot(p0[np.isfinite(p0)])
p1=np.nanmean(data1,1)
plot(p1[np.isfinite(p1)])

#%%
figure()
plot(im1.mean(1)[np.isfinite(p1)]-im1.mean(1)[np.isfinite(p1)].max())
plot(p1[np.isfinite(p1)]-np.nanmax(p1))
#%%
data0[np.isnan(data0)]=0
data1[np.isnan(data1)]=0
a0=ir.orientation_angle(data0)
a1=ir.orientation_angle(data1)
print('angle 0:',a0,'angle 1:',a1,' (0 is down)')

#%%
"""


img = cv2.medianBlur(cr.uint8sc(im0),11)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV,101,
                            np.uint8(np.std(img)))
kernel=np.ones((5,5))
#th3= cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
close(5)
close(6)
figure(5)
plt.hist(img[img>0], 256)
figure(6)
imshow(th3)
figure(7)
imshow(img)
imshow(th3, alpha=0.5)
figure(8)
imshow(im0)
#"""