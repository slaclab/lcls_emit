import numpy as np
import scipy.io
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from fitting_methods import *

class Image:
    def __init__(self, image, ncol, nrow, bg_image = None):
        self.ncol = ncol
        self.nrow = nrow
        self.flat_image = image
        self.bg_image = bg_image
        self.offset = 20
        
        self.proc_image = None
        self.x_proj = None
        self.y_proj = None
        self.xrms = None
        self.yrms = None
        self.xrms_error = None
        self.yrms_error = None
        self.xcen = None
        self.ycen = None
        self.xcen_error = None
        self.ycen_error = None
        self.xamp = None
        self.yamp = None
        self.xamp_error = None
        self.yamp_error = None
                
    def reshape_im(self, im = None):
        """Reshapes flattened OTR image to 2D array"""
        self.proc_image = self.flat_image.reshape(self.ncol,self.nrow)
        return self.proc_image
    
    def subtract_bg(self):
        """Subtracts bg image"""
        if self.bg_image is not None:
            self.bg_image = self.bg_image.reshape(self.ncol,self.nrow)
            if self.proc_image.shape == self.bg_image.shape:
                self.proc_image = self.proc_image - self.bg_image
        return self.proc_image

    def get_im_projection(self, subtract_baseline=True):
        """Expects ndarray, return x (axis=0) or y (axis=1) projection"""
        self.x_proj = np.sum(self.proc_image, axis=0)
        self.y_proj = np.sum(self.proc_image, axis=1)
        if subtract_baseline:
            self.x_proj = self.x_proj - np.mean(self.x_proj[0:self.offset])
            self.y_proj = self.y_proj - np.mean(self.y_proj[0:self.offset])
        return self.x_proj, self.y_proj    
            
    def dispatch(self, name, *args, **kwargs):
        fit_type_dict = {
            "gaussian": fit_gaussian_linear_background,
            "rms cut area": find_rms_cut_area
            }
        return fit_type_dict[name](*args, **kwargs)

    def get_sizes(self, method = "gaussian", show_plots = False, cut_area = 0.05):
        """Takes an image (2D array) and optional bg image, finds x and y projections,
        and fits with desired method. Current options are "gaussian" or "rms cut area".
        Returns size in x, size in y, error on x size, error on  y size"""
        
        # Find statistics
        para_x, para_error_x = self.dispatch(method, self.x_proj, para0=None, cut_area=cut_area, show_plots=show_plots)
        para_y, para_error_y = self.dispatch(method, self.y_proj, para0=None, cut_area=cut_area, show_plots=show_plots)
        
        self.xamp, self.yamp, self.xamp_error, self.yamp_error = \
        para_x[0],  para_y[0], para_error_x[0], para_error_y[0]

        self.xcen, self.ycen, self.xcen_error, self.ycen_error = \
        para_x[1],  para_y[1], para_error_x[1], para_error_y[1]

        #      size in x, size in y, error on x size, error on  y size
        self.xrms, self.yrms, self.xrms_error, self.yrms_error = \
        para_x[2],  para_y[2], para_error_x[2], para_error_y[2]
     
        
        return self.xrms, self.yrms, self.xrms_error, self.yrms_error, self.xamp, self.yamp 