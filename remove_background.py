# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:10:22 2016

@author: quentinpeter
"""
##############REPLACE NAMES HERE############################
backgroundfn='/Users/quentinpeter/Desktop/bg_resistance.tif'
imagefn='/Users/quentinpeter/Desktop/6p4uM_ovalbumin_resistance.tif'
outputfn='output.tif'

#the percentile is the approximate area of the image covered by background
percentile=100
blur=False
############################################################

import matplotlib.image as mpimg
import background_rm as rmbg
from PIL import Image
import cv2

bg=mpimg.imread(backgroundfn)
im=mpimg.imread(imagefn)

output=rmbg.remove_curve_background(im,bg,percentile=percentile)

if blur:
    output=cv2.GaussianBlur(output,(3,3),0)
    
outim = Image.fromarray(output)
outim.save(outputfn)
