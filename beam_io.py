# SETUP FILE FOR LCLS CU INJECTOR
import numpy as np
import sys, os, errno
import json
import time
import datetime
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
use_roi = False
ymin = 100
ymax = 350
xmin = 200
xmax = -1

# OTR2 resolution: MD update 10/22
resolution = 12.23*1e-6 # in meters for emittance calc

# Bad beam amp and sigma limits
# TODO: convert these to um for more intuitive setup?
amp_threshold = 1500 
min_sigma = 1.5 # noise
max_sigma = 40 # large/diffuse beam
max_samples = 3 # how many times to sample bad beam

# bg image subtraction
subtract_bg = False
bg_im = None
##################################

## EPICS I/O
# get PV info from json file (not comprehensive yet)
# TODO: make json file include all and easy to use/switch
meas_pv_info = json.load(open('meas_pv_info.json'))

im_pv = PV(meas_pv_info['diagnostic']['pv']['image'])
n_col_pv =  PV(meas_pv_info['diagnostic']['pv']['ncol'])
n_row_pv =  PV(meas_pv_info['diagnostic']['pv']['nrow'])

meas_input_pv =  PV(meas_pv_info['meas_device']['pv']['cntrl'])


x_size_pv = PV(meas_pv_info['diagnostic']['pv']['profmonxsize'])
y_size_pv = PV(meas_pv_info['diagnostic']['pv']['profmonysize'])

energy = caget(meas_pv_info['beam_info']['energy'])

opt_pv_info = json.load(open('opt_pv_info.json'))

opt_pvs = opt_pv_info['opt_vars']

savepaths = json.load(open('savepaths.json'))

#varx_pv =  PV(pv_info['device']['SOL']['SOL121'])
#vary_pv =  PV(pv_info['device']['QUAD']['Q121'])
#varz_pv =  PV(pv_info['device']['QUAD']['Q122'])

#varx_act_pv =  PV("SOLN:IN20:121:BACT")
#vary_act_pv =  PV("QUAD:IN20:121:BACT")
#varz_act_pv =  PV("QUAD:IN20:122:BACT")
#quad_act_pv =  PV("QUAD:IN20:525:BACT")


# resolution is now set up at top of this file 
# until a more robust solution is found
#resolution = caget(pv_info['device']['OTR2']['resolution'])*1e-6 # in meters for emittance calc

## I/O FUNCTIONS
#def get_beamsize_inj(set_list_pv,set_list_values, quad=meas_input_pv.get(), use_profMon=False):
#    """Get beamsize fn that changes upstream cu injector
#    and returns xrms and yrms in [m]"""
#    setinjector(varx,vary,varz)
#    beamsize = get_updated_beamsizes(quad, use_profMon=use_profMon)
#    return np.array([beamsize[0], beamsize[1]])

#def setinjector(set_list):
#    varx_pv.put(varx)
#    vary_pv.put(vary)
#    varz_pv.put(varz)
    
def setquad(value):
    """Sets Q525 to new scan value"""
    meas_input_pv.put(value)
    
def saveimage(im, ncol, nrow, beamsizes):
    """Saves images with col,row info and corresp. settings"""
    mkdir_p("saved_images")
    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    np.save(savepaths['images']+f'img_{timestamp}.npy', im)
    np.save(f'./saved_images/ncol_{timestamp}.npy', ncol)
    np.save(f'./saved_images/nrow_{timestamp}.npy', nrow)
    
    f= open(f"image_acq_quad_info.csv", "a+")
    bact = quad_act_pv.get()
    x_size = x_size_pv.get()
    y_size = y_size_pv.get()
    f.write(f"{timestamp},{ncol},{nrow},{xmin},{xmax},{ymin},{ymax},{resolution},{bact},{x_size},{y_size},{beamsizes[0]},{beamsizes[1]},{beamsizes[2]},{beamsizes[3]}\n")
    f.close()
    
def getbeamsizes(avg_num_images=3):
    """Returns xrms, yrms, xrms_err, yrms_err"""
#     if avg_num_images>1:
#         return average_ims(num_images = avg_num_images)
    beamsizes = average_ims(num_images = avg_num_images)
    return beamsizes 

def average_ims(num_images):
    xrms, yrms, xrms_err, yrms_err, xamp, yamp = [0]*num_images, [0]*num_images, [0]*num_images, \
    [0]*num_images, [0]*num_images, [0]*num_images
    
    for i in range(0,num_images):
        repeat = True
        count = 0
        # retake bad images 3 times
        while repeat:
            xrms[i], yrms[i], xrms_err[i], yrms_err[i], xamp[i], yamp[i] = get_beam_image()
            time.sleep(0.5)
            count = count + 1           

            if xamp[i]>amp_threshold and yamp[i]>amp_threshold and xrms[i]>min_sigma and yrms[i]>min_sigma and xrms[i]<max_sigma and yrms[i]<max_sigma:
                # if conditions are met, stop resampling this image
                repeat = False
            elif count==3:
                # if still bad after 3 tries, return 0
                xrms[i], yrms[i], xrms_err[i], yrms_err[i], xamp[i], yamp[i] = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
                repeat = False
       
    # only take ims w/ non-zero beam properties
    idx = ~np.isnan(xrms)
    if True not in idx:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    mean_xrms = np.mean(np.array(xrms)[idx])
    mean_yrms = np.mean(np.array(yrms)[idx])
    mean_xrms_err = np.std(np.array(xrms)[idx])/np.sqrt(num_images)
    mean_yrms_err = np.std(np.array(yrms)[idx])/np.sqrt(num_images)
#     mean_xrms_err = np.sqrt(np.mean(np.array(xrms_err)[idx]**2))
#     mean_yrms_err = np.sqrt(np.mean(np.array(yrms_err)[idx]**2))
    mean_xamp = np.mean(np.array(xamp)[idx])
    mean_yamp = np.mean(np.array(yamp)[idx]) 
    return mean_xrms, mean_yrms, mean_xrms_err, mean_yrms_err, mean_xamp, mean_yamp

def get_beam_image():
    """Get beam image from screen and return beamsize info"""
    im = im_pv.get()
    # average multiple images to obtain final image
#     if avg_num_images>1:
#         for i in range(avg_num_images):
#             im_tmp = im_pv.get()
#             im = np.mean(np.array([ im,im_tmp ]), axis=0 )
    ncol, nrow = n_col_pv.get(), n_row_pv.get()
    
    beam_image = Image(im, ncol, nrow, bg_image = bg_im)
    beam_image.reshape_im()
    if subtract_bg:
        beam_image.subtract_bg()
    if use_roi:
        beam_image.proc_image = beam_image.proc_image[ymin:ymax, xmin:xmax]
    beam_image.get_im_projection()
    
    # fit the profile and return the beamsizes
    beamsizes = beam_image.get_sizes(show_plots=False)
    save_beam = list(np.array(beamsizes[0:4])*resolution/1e-6)
    saveimage(im, ncol, nrow, beamsizes) # pass beamsizes in um
    return beamsizes

def get_updated_beamsizes(quad=quad_act_pv.get(), use_profMon=False, reject_bad_beam=True):
    """Get size should take a quad B field in kG and return 
    [xrms, yrms, xrms_err, yrms_err] in meters"""
    #setquad(quad)
    time.sleep(3)
    
    #use_profMon=True
    if use_profMon:
        xrms, xrms_err = x_size_pv.get()*1e-6, 0 # in meters
        yrms, yrms_err = y_size_pv.get()*1e-6, 0 # in meters
    else:
        beamsizes = np.array(getbeamsizes())
        #print(beamsizes)

        xrms = beamsizes[0]
        yrms = beamsizes[1]
        xrms_err = beamsizes[2]
        yrms_err = beamsizes[3]
        xamp = beamsizes[4]
        yamp = beamsizes[5]        

        if reject_bad_beam:
            count = 0
            while xamp<=amp_threshold or yamp<=amp_threshold or xrms<=min_sigma or yrms<=min_sigma or xrms>max_sigma or yrms>max_sigma or np.isnan(beamsizes).any():
               
                if count == 3:
                    # resample beamsize only 3 times
                    return np.nan, np.nan, np.nan, np.nan
                # TODO: switch to using the area under the gaussian (should be prop to charge)
                print("Low beam intensity/noisy or beam too small/large.")
                print("Waiting 5 sec and repeating measurement...")
                time.sleep(5)
                beamsizes = np.array(getbeamsizes())
                xrms = beamsizes[0]
                yrms = beamsizes[1]
                xrms_err = beamsizes[2]
                yrms_err = beamsizes[3]
                xamp = beamsizes[4]
                yamp = beamsizes[5] 
                
                count = count + 1

        # convert to meters
        xrms = xrms*resolution 
        yrms = yrms*resolution 
        xrms_err = xrms_err*resolution
        yrms_err = yrms_err*resolution 

    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    f= open(f"beamsize_config_info.csv", "a+")
    varx_cur = varx_act_pv.get()
    vary_cur = vary_act_pv.get()
    varz_cur = varz_act_pv.get()
    bact_cur = quad_act_pv.get()
    f.write(f"{timestamp},{varx_cur},{vary_cur},{varz_cur},{bact_cur},{xrms},{yrms},{xrms_err},{yrms_err}\n")
    f.close()
    return xrms, yrms, xrms_err, yrms_err


def mkdir_p(path):
    """Set up dirs for results in working dir"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise