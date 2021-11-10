# SETUP FILE FOR LCLS CU INJECTOR
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
    from epics import caget, caput, PV
except:
    print("did not import epics")
from image import Image

##################################
# SET UP AT BEGINNING OF SHIFT 
##################################
# ROI: y is col, x is row
use_roi = True
ymin = 200
ymax = 900
xmin = 1000
xmax = -1

# Bad beam amp and sigma limits
# TODO: convert these to um for more intuitive setup?
amp_threshold = 1500 
min_sigma = 1.5 # noise
max_sigma = 25  # large/diffuse beam

# OTR2 resolution: MD update 10/22
resolution = 12.23*1e-6 # in meters for emittance calc

# bg image subtraction
subtract_bg = False
bg_im = None
##################################

## EPICS I/O
# get PV info from json file (not comprehensive yet)
# TODO: make json file include all and easy to use/switch
pv_info = json.load(open('pv_info.json'))

im_pv = PV("OTRS:IN20:571:IMAGE")
n_col_pv =  PV("OTRS:IN20:571:ROI_YNP")
n_row_pv =  PV("OTRS:IN20:571:ROI_XNP")

meas_input_pv =  PV(pv_info['device']['QUAD']['Q525'])
quad_act_pv =  PV("QUAD:IN20:525:BACT")
varx_pv =  PV(pv_info['device']['SOL']['SOL121'])
vary_pv =  PV(pv_info['device']['QUAD']['Q121'])
varz_pv =  PV(pv_info['device']['QUAD']['Q122'])

x_size_pv = PV(pv_info['device']['OTR2']['profmonxsize'])
y_size_pv = PV(pv_info['device']['OTR2']['profmonysize'])

energy = caget(pv_info['energy']['DL1'])
# resolution is now set up at top of this file 
# until a more robust solution is found
#resolution = caget(pv_info['device']['OTR2']['resolution'])*1e-6 # in meters for emittance calc

## I/O FUNCTIONS
def setquad(value):
    """Sets Q525 to new scan value"""
    meas_input_pv.put(value)
    
def saveimage(im, ncol, nrow, beamsizes):
    """Saves images with col,row info and corresp. settings"""
    mkdir_p("saved_images")
    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    np.save(f'./saved_images/img_{timestamp}.npy', im)
    np.save(f'./saved_images/ncol_{timestamp}.npy', ncol)
    np.save(f'./saved_images/nrow_{timestamp}.npy', nrow)
    
    f= open(f"image_acq_quad_info.csv", "a+")
    bact = quad_act_pv.get()
    x_size = x_size_pv.get()
    y_size = y_size_pv.get()
    f.write(f"{timestamp},{ncol},{nrow},{resolution},{bact},{x_size},{y_size},{beamsizes[0]},{beamsizes[1]}\n")
    f.close()
    
def getbeamsizes(avg_num_images=1):
    """Returns xrms, yrms, xrms_err, yrms_err"""
    im = im_pv.get()
    # average multiple images to obtain final image
    if avg_num_images>1:
        for i in range(avg_num_images):
            im_tmp = im_pv.get()
            im = np.mean(np.array([ im,im_tmp ]), axis=0 )
        
    ncol, nrow = n_col_pv.get(), n_row_pv.get()
    
    beam_image = Image(im, ncol, nrow, bg_image = bg_image)
    beam_image.reshape_im()
    if subtract_bg:
        beam_image.subtract_bg()
    if use_roi:
        beam_image.proc_image = beam_image.proc_image[ymin:ymax, xmin:xmax]
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
    """Set up dirs for results in working dir"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
 