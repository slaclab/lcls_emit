import numpy as np
import scipy.io
import sys
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian_linear_background(x, amp, mu, sigma, slope=0, offset=0):
    """Gaussian plus linear background fn"""
    return amp*np.exp( -(x-mu)**2/2/sigma**2 ) + slope * x + offset 

def fit_gaussian_linear_background(y, para0 = None, show_plots=False, cut_area = None):
    """Takes a function y and inputs and fits and Gaussian with
    linear bg to it. Returns the best fit estimates of the parameters 
    amp, mu, sigma and their associated 1sig error """
    from scipy.optimize import curve_fit

    x = np.arange(y.shape[0])

    if para0 == None:
        offset0 =  np.mean(y[-10:]) #y.min()
        amp0 = y.max() - offset0
        mu0 = x[np.argwhere(y==y.max())][0].item() # get the first element if more than two 
        try:
            sigma0 = (x[np.argwhere(y>np.exp(-0.5*2**2)*amp0+offset0).max()]\
                      - x[np.argwhere(y>np.exp(-0.5*2**2)*amp0+offset0).min()] )/(2*2)
        except:
            sigma0 = 5
        slope0 = 0
        para0 = [amp0, mu0, sigma0, slope0, offset0]

    try:
        para, para_error = curve_fit(gaussian_linear_background, x, y, p0 = para0)
    except:
        print("Fitting failed.")
        para = para0 
        para_error = [0]*len(para0)

    para[2] = abs(para[2])  # Gaussian width is postivie definite
    # contraints on the output fit parameters
    if para[2] >= len(x):
        para[2] = len(x)

    if abs(para[1]) <= 0:
        para[1] = 0

    if abs(para[1]) >= len(x):
        para[1] = len(x)
        
    plot_fit(x, y, para, show_plots=show_plots)

    return para[0:3], np.sqrt(np.diag(para_error))[0:3] 

def find_rms_cut_area(y, para0 = None, show_plots=False, cut_area = 0.05):
    """Takes a distribution (ndarray) and the desired cut area (5% is default). 
    Returns the amp (max of array), mean of distribution, and rms (std) of dist"""
    x = np.arange(y.shape[0])
    y = np.array([0 if ele < 0 else ele for ele in y])

    cumsum = np.cumsum(y)
    idLow = int(np.argwhere(cumsum < cut_area/2*cumsum[-1])[-1])
    idHigh = int(np.argwhere(cumsum > (1-cut_area/2)*cumsum[-1])[0])

    y[0:idLow] = y[idHigh:]=0

    xx = x[y != 0] 
    xp = y[y != 0]

    mean = np.sum(xx*xp)/ np.sum(xp)
    mean2 = np.sum(xx*xx*xp)/np.sum(xp)
    var = mean2 - mean**2
    std = np.sqrt(var)

    # TODO: better estimate of peak amplitude in case of noise
    amp = max(y)

    para = np.array([amp, mean, std])

    # TODO: implement errors
    para_errors = np.array([0]*len(para))
    
    if show_plots:
        plot_fit(x, y, para)

    return para, para_errors

def plot_fit(x, y, para_x, show_plots=True):
    timestamp = (datetime.datetime.now()).strftime("%m-%d_%H-%M-%S")
    fig = plt.figure(figsize=(7 ,5))
    plt.plot(x, y, 'b-', label='data')
    plt.plot(x, gaussian_linear_background(x, *para_x), 'r-', label='fit: amp=%f, centroid=%f, sigma=%f' % tuple(para_x[:3]))
    plt.xlabel("Pixel")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(f"./plots/beamsize_fit_{timestamp}.png")
    if show_plots:
        plt.show()
    