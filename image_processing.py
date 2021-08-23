import numpy as np
import scipy.io
import sys

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def im_projection(image, axis=0, subtract_baseline=True):
    """Expects ndarray, return x (axis=0) or y (axis=1) projection"""
    proj = np.sum(image, axis)
    if subtract_baseline:
        return proj - min(proj)
    return proj

def gaussian_linear_background(x, amp, mu, sigma, slope, offset):
    return amp*np.exp( -(x-mu)**2/2/sigma**2 ) + slope * x + offset 

def fit_gaussian_linear_background_v2(x, y, para0 = None):
    if para0 == None:
        offset0 =  np.mean(y[-5:]) #y.min()
        amp0 = y.max() - offset0
        mu0 = x[np.argwhere(y==y.max())][0].item() # get the first element if more than two 
        try:
            sigma0 = (x[np.argwhere(y>np.exp(-0.5*2**2)*amp0+offset0).max()] - x[np.argwhere(y>np.exp(-0.5*2**2)*amp0+offset0).min()] )/(2*2)
        except:
            sigma0 = 5
        slope0 = 0
        para0 = [amp0, mu0, sigma0, slope0, offset0]
        
    try:
        para, para_error = curve_fit(gaussian_linear_background, x, y, p0 = para0)
    except:
        print("Fitting failed.")
        para = para0 
        para_error = []
    
    para[2] = abs(para[2])  # Gaussian width is postivie definite
    # contraints on the output fit parameters
    if para[2] >= len(x):
        para[2] = len(x)
        
    if abs(para[1]) <= 0:
        para[1] = 0
        
    if abs(para[1]) >= len(x):
        para[1] = len(x)
    return para, np.sqrt(np.diag(para_error_x)) #error is std

def image_analysis(image, bg_image=None):
    if bg_image is not None and image.shape==bg_image.shape:
        image = image - bg_image

    x_proj = im_projection(image, axis=0)
    y_proj = im_projection(image, axis=1)

    xx = np.arange(x_proj.shape[0])
    yy = np.arange(y_proj.shape[0])

    # Gaussian fit
    para_x, para_error_x = fit_gaussian_linear_background_v2(xx, x_proj, para0 = None)
    para_y, para_error_y = fit_gaussian_linear_background_v2(yy, y_proj, para0 = None)
    
    # TODO: Add optional plotting and save plots of fitting
    
    return para_x[2], para_error_x[2], para_y[2], para_error_y[2]