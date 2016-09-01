# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:10:22 2016

@author: quentinpeter

Plot of the 12 points of the diffusion device

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


#load im files names
imagefn=    '../Data/Maya_images/im_*.tif'
backgroundfn='../Data/Maya_background/*.tif'

#imports everithing needed
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import background_rm as rmbg
from glob import glob
import numpy as np
from natsort import natsorted
import scipy
import matplotlib
import warnings

gfilter=scipy.ndimage.filters.gaussian_filter1d

#Get color for plot
cmap = matplotlib.cm.get_cmap('Spectral')

#Get list data
imagefn=natsorted(glob(imagefn))
backgroundfn=natsorted(glob(backgroundfn))

#load images
bgs=[mpimg.imread(fn) for fn in backgroundfn]
imgs=[mpimg.imread(fn) for fn in imagefn]

assert(len(bgs)==len(imgs))
vari=[]
for i, im in enumerate(imgs):
    #remove background
    output=rmbg.remove_curve_background(im,bgs[i],method='detectChannel', 
                                        xOrientate=True)
    #get profile and center with maximum of filtered
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        fil=gfilter(np.nanmean(output,1)[200:800],21)
        amax=fil.argmax()
        profile=np.nanmean(output,1)[200:800]

    #Compute X and restrict to a 300px wide channel
    X=np.arange(600)-amax
    valid=np.logical_and(X>-150,X<150)
    X=X[valid]
    Y=profile[valid]

    #Get var and save normalized variance
    valmax=Y.max()
    #.05 Choosen to get rid of dust but may be changed
    var=np.nansum((Y*X**2)[Y>.05])/np.nansum(Y[Y>.05]) 
    valmax=np.nansum(Y[Y>0])/np.sqrt(2*np.pi*var)
    #The "Normalized" variance is var/a, with function f=a*norm(0,var)
    #As Y.max is a/sqrt(2 pi var):
    vari.append(var/valmax/np.sqrt(2*np.pi*var))
    
    print(np.nansum(Y[Y>0]),valmax*np.sqrt(2*np.pi*var))
    
    #plot profile and fit
    plt.plot(X,Y-.1*i,c=cmap(i/len(imgs)))
    plt.plot(X,valmax*np.exp(-(X**2)/(2*var))-.1*i,c=cmap(i/len(imgs)))

#%% plot variances
X=np.array([3.5,5.3,8.6,10.3,18.6,20.4,28.6,30.4,58.7,60.5,88.7,90.5])
Y=vari[1:]
plt.figure()
plt.plot(X,Y,'x')
plt.plot(X,X*(X*Y).sum()/(X*X).sum())
plt.xlabel('Distance [mm]')
plt.ylabel('Variance')







