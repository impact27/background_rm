from unittest import TestCase

import background_rm as rmbg
from tifffile import imread
import numpy as np
from . import __file__ 
import os
folder = os.path.dirname(__file__)


class TestImage(TestCase):
    def test_registration_UV(self):
        bg = imread(folder + '/test_data/bg.tif')
        im0 = imread(folder + '/test_data/im.tif')
        data0=rmbg.remove_curve_background(im0,bg)
        self.assertTrue(np.nanstd(np.nanmean(im0, 0)/np.mean(im0)-1) > 
                        np.nanstd(np.nanmean(data0, 0)))
        
