import numpy as np
import sys, os, errno
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

im_pv = "OTRS:IN20:571:IMAGE"
n_col_pv = "OTRS:IN20:571:ROI_YNP"
n_row_pv = "OTRS:IN20:571:ROI_XNP"

meas_input = pv_info['device']['QUAD']['Q525']
quad_act = "QUAD:IN20:525:BACT"
varx_pv = pv_info['device']['SOL']['SOL121']
vary_pv = pv_info['device']['QUAD']['Q121']
varz_pv = pv_info['device']['QUAD']['Q122']

x_size_pv = pv_info['device']['OTR2']['profmonxsize']
y_size_pv = pv_info['device']['OTR2']['profmonysize']

energy = caget(pv_info['energy']['DL1'])
#resolution = caget(pv_info['device']['OTR2']['resolution'])*1e-6 # in meters for emittance calc
# MD update 10/22
resolution = 12.23*1e-6

def setquad(value):
    """Sets Q525 to new scan value"""
    caput(meas_input, value)
    
def saveimage(im, ncol, nrow, beamsizes):
    mkdir_p("saved_images")
    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    np.save(f'./saved_images/img_{timestamp}.npy', im)
    np.save(f'./saved_images/ncol_{timestamp}.npy', ncol)
    np.save(f'./saved_images/nrow_{timestamp}.npy', nrow)
    
    f= open(f"image_acq_quad_info.csv", "a+")
    bact = caget(quad_act)
    x_size = caget(x_size_pv)
    y_size = caget(y_size_pv)
    f.write(f"{timestamp},{ncol},{nrow},{resolution},{bact},{x_size},{y_size},{beamsizes[0]},{beamsizes[1]}\n")
    f.close()
    
def getbeamsizes(avg_num_images=1):
    """Returns xrms, yrms, xrms_err, yrms_err"""
    im = caget(im_pv)
    # average multiple images to obtain final image
    if avg_num_images>1:
        for i in range(avg_num_images):
            im_tmp = caget(im_pv)
            im = np.mean(np.array([ im,im_tmp ]), axis=0 )
        
    ncol, nrow = caget(n_col_pv), caget(n_row_pv)
    
    beam_image = Image(im, ncol, nrow, bg_image = None)
    beam_image.reshape_im()
    beam_image.subtract_bg()
    #beam_image.proc_image = beam_image.proc_image[200:900, 1000:]
    beam_image.get_im_projection()
    
    # fit the profile and return the beamsizes
    beamsizes = beam_image.get_sizes(show_plots=False)

    saveimage(im, ncol, nrow, beamsizes*resolution/1e-6) # pass beamsizes in um
    return beamsizes 

def get_updated_beamsizes(quad):
    """Get size should take a quad B field in kG and return [xrms, yrms] in meters"""
#     setquad(quad)
#     time.sleep(3)
    beamsizes = np.array(getbeamsizes())[0:2]*resolution # convert to meters
    xrms = beamsizes[0]
    yrms = beamsizes[1]
    return xrms, yrms

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
 