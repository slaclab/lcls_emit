import numpy as np
import sys
import json
import time
import datetime

import scipy.io
import scipy.ndimage as snd
from scipy.stats import moment
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
try:
    from epics import caget, caput
except:
    print("did not import epics")
    
from image import Image

# get PV info
pv_info = json.load(open('pv_info.json'))

im_pv = pv_info['device']['OTR2']['image']
n_col_pv = pv_info['device']['OTR2']['ncol']
n_row_pv = pv_info['device']['OTR2']['nrow']

meas_input = pv_info['device']['QUAD']['Q525']
varx_pv = pv_info['device']['SOL']['SOL121']
vary_pv = pv_info['device']['QUAD']['Q121']
varz_pv = pv_info['device']['QUAD']['Q122']

energy = caget(pv_info['energy']['DL1'])
resolution = caget(pv_info['device']['OTR2']['resolution'])*1e-6


def setquad(value):
    """Sets Q525 to new scan value"""
    caput(meas_input, value)
    
def saveimage(im, ncol, nrow):
    timestamp = (datetime.datetime.now()).strftime("%m-%d_%H-%M-%S")
    np.save(f'./saved_images/img_{timestamp}.npy', im)
    np.save(f'./saved_images/ncol_{timestamp}.npy', ncol)
    np.save(f'./saved_images/nrow_{timestamp}.npy', nrow)
    
def getbeamsizes():
    """Returns xrms, yrms, xrms_err, yrms_err"""
    im, ncol, nrow = caget(im_pv), caget(n_col_pv), caget(n_row_pv)
    
    saveimage(im, ncol,nrow)
    
    beam_image = Image(im, ncol, nrow, bg_image = None)
    beam_image.reshape_im()
    beam_image.subtract_bg()
    beam_image.get_im_projection()
    # fit the profile and return the beamsizes
    return beam_image.get_sizes(show_plots=True)

def get_sizes(quad):
    """Get size should take a quad B field in kG and return [xrms, yrms] in meters"""
    setquad(quad)
    time.sleep(3)
    beamsizes = getbeamsizes()[0:2]*resolution # convert to meters
    xrms = beamsizes[0]
    yrms = beamsizes[1]
    return xrms, yrms
    