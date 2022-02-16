
    
import  errno

from image import Image
from os.path import exists
from epics import caget,  PV
import epics

def isotime():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()
isotime()
    
import numpy as np

from fitting_methods import *



############ SETUP ######################
rootp = '/home/fphysics/edelen/sw/lcls_emit/'

im_proc = json.load(open(rootp+'config_files/img_proc.json'))
meas_pv_info = json.load(open(rootp+'config_files/meas_pv_info.json'))
pv_savelist = json.load(open(rootp+'config_files/save_scalar_pvs.json'))
opt_pv_info = json.load(open(rootp+'config_files/opt_pv_info.json'))
savepaths = json.load(open(rootp+'config_files/savepaths.json'))

online=False
##################################

def mkdir_p(path):
    """Set up dirs for results in working dir"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

mkdir_p(savepaths['images'])
mkdir_p(savepaths['summaries'])
mkdir_p(savepaths['fits'])

opt_pvs = opt_pv_info['opt_vars']

subtract_bg = im_proc['subtract_bg']
bg_image = im_proc['background_im']
use_roi = im_proc['use_roi']
roi_xmin = im_proc['roi']['xmin']
roi_ymin = im_proc['roi']['ymin']
roi_xmax = im_proc['roi']['xmax']
roi_ymax = im_proc['roi']['ymax']
avg_ims = im_proc['avg_ims']
n_acquire = im_proc['n_to_acquire']
amp_threshold_x = im_proc['amp_threshold_x']#100 #1500 
amp_threshold_y = im_proc['amp_threshold_y']# 200
min_sigma = im_proc['min_sigma']#2.0##1.5 # noise
max_sigma = im_proc['max_sigma']#800##40 # large/diffuse beam
max_samples = im_proc['max_samples']#3 # how many times to sample bad beam

resolution = epics.caget(meas_pv_info['diagnostic']['pv']['resolution'])*10**-6 # in meters for emittance calc


if online:

    im_pv = PV(meas_pv_info['diagnostic']['pv']['image'])
    n_col_pv =  PV(meas_pv_info['diagnostic']['pv']['ncol'])
    n_row_pv =  PV(meas_pv_info['diagnostic']['pv']['nrow'])
    x_size_pv = PV(meas_pv_info['diagnostic']['pv']['profmonxsize'])
    y_size_pv = PV(meas_pv_info['diagnostic']['pv']['profmonysize'])

    meas_cntrl_pv =  PV(meas_pv_info['meas_device']['pv']['cntrl'])
    meas_read_pv =  PV(meas_pv_info['meas_device']['pv']['read'])


### MAKE INITIAL FILES FOR CSV #########

file_exists = exists(savepaths['summaries']+"image_acq_quad_info.csv")

#print('fe',file_exists)

if not file_exists:
    #print('foo1')

    #todo add others as inputs
    f= open(savepaths['summaries']+"image_acq_quad_info.csv", "a+")
    f.write(f"{'timestamp'},{'ncol'},{'nrow'},{'roi_xmin'},{'roi_xmax'},{'roi_ymin'},{'roi_ymax'},{'resolution'},{'bact'},{'x_size'},{'y_size'},{'beamsizes[0]'},{'beamsizes[1]'},{'beamsizes[2]'},{'beamsizes[3]'}\n")
    f.close()
    

file_exists = exists(savepaths['summaries']+"beamsize_config_info.csv")
#print('fe',file_exists)
if not file_exists:
    #print('foo2')
    #todo add others as inputs
    f= open(savepaths['summaries']+"beamsize_config_info.csv", "a+")
    f.write(f"{'timestamp'},{'varx_cur'},{'vary_cur'},{'varz_cur'},{'bact_cur'},{'xrms'},{'yrms'},{'xrms_err'},{'yrms_err'}\n")
    f.close()
    
    

#########################    
#########################    
       
        
class Screen:
    def __init__(self,  ):

        
        self.save_image_directory = save_image_directory
        self.show_plots_flag = show_plots_flag
        
        self.n_samples = n_samples
        self.save_images_flag = save_images_flag
        self.average_measurements_flag = average_measurements_flag
        self.use_ROI_flag = use_ROI_flag
        self.subtract_bg_flag = subtract_bg_flag
    
    
        self.resolution
        self.resolutionPV
        self.ncolPV = 
        self.nrowPV = 
        self.imgPV = 
        self.resolution_factor =10**-6 #to meters

        
        self.roi = None
        #[roi xmin, roi xmax, roi ymin, roi ymax]
    
        self.reject_bad
        self.min_sigma
        self.max_sigma
        self.amp_threshold_x
        self.amp_threshold_y

        
        self.beam_image = None
        self.img = None
        self.ncol = None
        self.nrow = None
        self.xrms
        self.yrms
        self.xrms_err
        self.yrms_err
        self.amp_x
        self.amp_y
        

    def get_beam_image(img_collect = None, self.ncol = None, self.nrow = None,reject_bad=self.reject_bad):
            
            
            if  img_collect == None:
                
                img_collect = [0]*self.n_samples  
                
                for i in range(0,self.n_samples):
                    img = self.imgPV.get()
                    self.ncol, self.nrow = self.ncolPV.get(), self.nrowPV.get()

                    beam_image = Image(img, ncol, nrow, self.bg_image)
                    beam_image.reshape_im()

                    if self.subtract_bg_flag:
                        beam_image.subtract_bg()

                    if self.use_ROI_flag:
                        beam_image.proc_image = beam_image.proc_image[self.roi[2]:self.roi[3], self.roi[0]:self.roi[1]]
                        
                        beam_image.get_im_projection()

                    img_collect[i] = beam_image.proc_image
        

            img = np.mean(img_collect, axis=0)
        
            img = Image(img, self.ncol, self.nrow, self.bg_image)
            
            img.reshape_im()
            img.get_im_projection()
            
            self.img = img.
        
            #plt.imshow(im.proc_image)
        
            beamsizes = img.get_sizes(show_plots=self.show_plots_flag)

            
            # fit the profile and return the beamsizes
            beamsizes = list(self.beam_image.get_sizes(show_plots=self.show_plots_flag))
            self.resolution = caget(self.resolutionPV)*self.resolution_factor
            self.xrms = beamsizes[0]*self.resolution
            self.yrms = beamsizes[1]*self.resolution
            self.xrms_err = beamsizes[2]*self.resolution
            self.yrms_err = beamsizes[3]*self.resolution
            self.amp_x = beamsizes[4]
            self.amp_y = beamsizes[5]
            #timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f") 
            #savesummary(beamsizes,timestamp)# pass beamsizes in um
            #saveimage(im, ncol, nrow, timestamp, avg_img=False)
        
            if reject_bad:
                reject_check(self.xrms,self.yrms,self.xrms_err,self.yrms_err,self.amp_x,self.amp_y)
                
            return [self.xrms,self.yrms,self.xrms_err,self.yrms_err]

        
    def reject_check(xrms, yrms,xrms_err,yrms_err,amp_x,amp_y):
                if self.amp_x<self.amp_threshold_x:
                    self.xrms = np.nan
                    print('beam rejected for low amplitude x ')
                    
                if self.amp_x<self.amp_threshold_y:
                    self.amp_y = np.nan
                    print('beam rejected for low amplitude y ')
                    
                if self.xrms<self.min_sigma:
                    self.xrms = np.nan
                    print('beam rejected for low xrms')
                    
                if self.yrms<self.min_sigma:
                    self.yrms = np.nan
                    print('beam rejected for low yrms')
                    
                    
                if self.xrms>self.max_sigma:
                    self.xrms = np.nan
                    print('beam rejected for large xrms')
                    
                if self.yrms>self.max_sigma:
                    self.yrms = np.nan
                    print('beam rejected for large yrms')
                    
    def get_beamsizes(use_profMon = self.use_profMon,reject_bad = self.reject_bad):
        if use_profMon:
                self.xrms, self.xrms_err = x_size_pv.get()*1e-6, 0.001 # in meters
                self.yrms, self.yrms_err = y_size_pv.get()*1e-6, 0.001 # in meters
                
        if not use_profMon:
            get_beam_image()
        
        return self.xrms, self.yrms, self.xrms_err, self.yrms_err
            
                
            
        


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
            try:
                self.bg_image = self.bg_image.reshape(self.ncol,self.nrow)
            except:
                "Error with background image and recorded image sizes"
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
        #self.x_proj = np.clip(self.x_proj,-500,np.inf)
        #self.y_proj = np.clip(self.y_proj,3000,np.inf)
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
    
    
 