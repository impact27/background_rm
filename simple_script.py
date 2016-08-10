# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:10:22 2016

@author: quentinpeter

Simple usage of the background removal script

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


##############REPLACE NAMES HERE############################
#image name goes through glob => use * if you want
imagefn='/Users/quentinpeter/Desktop/25uM_Transferrin_resistance.tif'
backgroundfn='/Users/quentinpeter/Desktop/bg_resistance.tif'
outputfn='output'

#the percentile is the approximate area of the image covered by background
percentile=None
blur=False
xOrientate=True
twoPass=True
############################################################

#imports everithing needed
import matplotlib.image as mpimg
import background_rm as rmbg
from PIL import Image
import cv2
from glob import glob
import numpy as np

def printInfo(d,m):
    print('mean',d[m].mean(),
          'std',d[m].std(),
          'std/SQRT(N)',d[m].std()/np.sqrt(m.sum()))

#load images
bg=mpimg.imread(backgroundfn)
imgs=[mpimg.imread(fn) for fn in glob(imagefn)]

for i, im in enumerate(imgs):
    #remove background
    output=rmbg.remove_curve_background(im,bg,percentile=percentile, 
                                        xOrientate=xOrientate, twoPass=twoPass)
    
    #blur if asked
    if blur:
        output=cv2.GaussianBlur(output,(3,3),0)
        
    #save image 
    outim = Image.fromarray(output,'F')
    outim.save(outputfn+str(i)+'.tif')
    
    sm=rmbg.signalMask(output,10,True)
    bm=rmbg.backgroundMask(output,10,True)
    
    print('Signal '+str(i))
    printInfo(output,sm)
    print('Background '+str(i))
    printInfo(output,bm)
    
  




