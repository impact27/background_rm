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
backgroundfn='/Users/quentinpeter/Desktop/bg_resistance.tif'
imagefn='/Users/quentinpeter/Desktop/6p4uM_ovalbumin_resistance.tif'
outputfn='output.tif'

#the percentile is the approximate area of the image covered by background
percentile=None
blur=False
############################################################

#imports everithing needed
import matplotlib.image as mpimg
import background_rm as rmbg
from PIL import Image
import cv2

#load images
bg=mpimg.imread(backgroundfn)
im=mpimg.imread(imagefn)

#remove background
output=rmbg.remove_curve_background(im,bg,percentile=percentile)

#blur if asked
if blur:
    output=cv2.GaussianBlur(output,(3,3),0)
    
#save image 
outim = Image.fromarray(output)
outim.save(outputfn)
