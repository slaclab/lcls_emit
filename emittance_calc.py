import datetime
import numpy as np 
import warnings
import sys, os, errno
import scipy
from scipy.optimize import curve_fit
# on sim 
#from beam_io_sim import get_sizes
# on lcls
from beam_io import get_beamsizes, setquad
get_sizes = get_beamsizes
import json
from os.path import exists

import epics
from epics import caget,caput
def isotime():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()
isotime()


# do not display warnings when cov can't be computed
# this will happen when len(y)<=3 and yerr=0
warnings.simplefilter('ignore', scipy.optimize.OptimizeWarning)

rootp = '/home/fphysics/edelen/sw/lcls_emit/'

beamline_info = json.load(open(rootp+'config_files/beamline_info.json'))

m_0 = beamline_info['m_0']
d = beamline_info['d']
l = beamline_info['l']
twiss0 = beamline_info['Twiss0']
energy = beamline_info['energy']

#load info about where to put saving of raw images and summaries; make directories if needed and start headings
savepaths = json.load(open(rootp+'config_files/savepaths.json'))

pv_savelist = json.load(open(rootp+'config_files/save_scalar_pvs.json'))

def mkdir_p(path):
    """Set up dirs for results in working dir"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

mkdir_p(savepaths['summaries'])

file_exists = exists(savepaths['summaries']+"emit_calc_log.csv")

if not file_exists:

    #todo add others as inputs
    f= open(savepaths['summaries']+"emit_calc_log.csv", "a+")
    f.write(f"{'timestamp'},{'nex'},{'ney'},{'bmx'},{'bmy'},{'xsizes'},{'ysizes'},{'kx'},{'ky'},{'adapted'}\n")
    f.close()

def func(x, a, b, c):
    """Polynomial function for emittance fit"""
    return a*x**2 + b*x + c

# function to create path to output if dir was not created before
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def get_gradient(b_field, l_eff=0.108):
    """Returns the quad field gradient [T/m] 
        l_eff: effective length [m] 
        b_field: integrated field [kG]"""
    return np.array(b_field)*0.1 /l_eff
    
def get_k1(g, p):
    """Returns quad strength [1/m^2]
       g: quad field gradient [T/m]
       p: momentum [GeV] (or alternatively beta*E [GeV])"""
    return 0.2998 * g / p

def fit_sigma(sizes, k, axis, sizes_err=None, d=d, l=l, adapt_ranges=False, num_points=5, show_plots=False):
    """Fit sizes^2 = c0 + c1*k + c2*k^2
       returns: c0, c1, c2"""
    sizes = np.array(sizes)
    if len(sizes)<3:
        print("Less than 3 data points were passed.")
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    if sizes_err is not None and sizes.all()>0 and sizes_err.all()>0:
        w = 2*sizes*np.array(sizes_err) # sigma for poly fit
        abs_sigma = True
    else:
        w = None
        abs_sigma = False
    coefs, cov = curve_fit(func, k, sizes**2, sigma=w, absolute_sigma=abs_sigma)
    
    if axis == 'x':
        min_k, max_k = np.min(k), np.max(k)
    elif axis == 'y':
        min_k, max_k = np.min(k), np.max(k)
    
    xfit = np.linspace(min_k, max_k, 100)
    

    # FOR DEBUGGING ONLY
    #plot_fit(k, sizes, xfit, yerr=sizes_err, axis=axis, save_plot=False, show_plots=show_plots)  
    
    if adapt_ranges:
        try:
            coefs, cov, k = adapt_range(k, sizes, w=w, axis=axis, fit_coefs=coefs, x_fit=xfit, energy=energy, num_points=num_points, save_plot=True, show_plots=show_plots)
            # log data
            timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
            if axis=="x":
                save_data(timestamp,0,0,0,0,0,0,0,0,sizes,0,k,0,str(adapt_ranges))
            if axis=="y":
                save_data(timestamp,0,0,0,0,0,0,0,0,0,sizes,0,k,str(adapt_ranges))
#         except NameError:
#             print("Error: A function to get beamsizes is not defined. Returning original fit.")
#             plot_fit(k, sizes, xfit, yerr=sizes_err, axis=axis, save_plot=True, show_plots=show_plots)
        except ComplexRootError:
            print("Error: Cannot adapt quad ranges. Returning original fit.")
            plot_fit(k, sizes, xfit, yerr=sizes_err, axis=axis, save_plot=True, show_plots=show_plots)
        except ConcaveFitError:
            print("Error: Cannot adapt quad ranges due to concave poly. Returning original fit.")
            plot_fit(k, sizes, yerr=sizes_err, axis=axis, save_plot=True, show_plots=show_plots)     
    else:
        plot_fit(k, sizes, xfit, yerr=sizes_err, axis=axis, save_plot=True, show_plots=show_plots)
  
    if np.isnan(coefs).any() or np.isnan(cov).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan 
        
    # poly.poly: return c0,c1,c2
    # np.polyfit: highest power first
    c2, c1, c0 = coefs
    coefs_err = np.sqrt(np.diag(cov))
    
    emit2 = (4*c0*c2 - c1**2) / l**2 / (4*d**4)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            emit = np.sqrt(emit2)
            # error propagation for dependent variables
            emit_gradient = 1./(4*l*d**2*emit) * np.array([[4*c0, -2*c1, 4*c2]]).T
            emit_err = np.sqrt(np.matmul(np.matmul(emit_gradient.T, cov), emit_gradient))
            return emit, emit_err[0][0], coefs, coefs_err, k
    except RuntimeWarning:
        return np.nan, np.nan, np.nan, np.nan, np.nan 
    
def propagate_sigma(sigma_mat2, mat2):
    return (mat2 @ sigma_mat2) @ mat2.T

def drift_mat2(L):
    return np.array( [ [1, L], [0, 1] ])

def quad_mat2(kL, L=l):
    """
    Quadrupole transfer matrix, 2x2. Note that 
    
    """
    
    if L == 0:
        return thin_quad_mat2(kL)
    
    k = kL/L
    
    if k == 0:
        mat2 = drift_mat2(L)
    elif k > 0:
        # Focusing
        rk = np.sqrt(k)
        phi = rk*L
        mat2 =  [ [np.cos(phi), np.sin(phi)/rk], [-rk*np.sin(phi), np.cos(phi)] ]
    else:
        # Defocusing
        rk = np.sqrt(-k)
        phi = rk*L
        mat2 = [ [np.cosh(phi), np.sinh(phi)/rk], [rk*np.sinh(phi), np.cosh(phi)] ]
        
    return mat2

def quad_drift_mat2(kL, *, Ltot=d+l,  Lquad=l):
    """
    Composite [quad, drift] 2x2 transfer matrix. 
    """
    
    Ldrift = Ltot - Lquad
    
    return drift_mat2(Ldrift) @ quad_mat2(kL, Lquad)

def get_bmag(coefs, coefs_err, k, emit, emit_err, axis, twiss0 = twiss0):
    """Calculates Bmag from calculated emittance
    and from initial Twiss at OTR2: HARDCODED from Matlab GUI"""
    # HARDCODED INIT TWISS PARAMS
    # TODO: clean up LCLS/machine specific info
    #twiss0 = [1e-6, 1e-6, 1.113081026, 1.113021659, -6.89403587e-2, -7.029489754e-2]
    
    c2, c1, c0 = coefs
    c2_err, c1_err, c0_err = coefs_err
    
    sig11 = c2 / (d*l)**2
    sig12 = (-c1 - 2*d*l*sig11) / (2*d**2*l)
    sig22 = (c0 - sig11 - 2*d*sig12) / d**2
    
    # Matrix form
    sigma0 = np.array([ [sig11, sig12], [sig12, sig22] ])
    
    beta0 =  twiss0[2] if axis == 'x' else twiss0[3] if axis == 'y' else 0
    alpha0 = twiss0[4] if axis == 'x' else twiss0[5] if axis == 'y' else 0
    gamma0 = (1+alpha0**2)/beta0

    beta_quad = sig11/emit
    alpha_quad = -sig12/emit
    gamma_quad = sig22/emit
    
    # Propagate forward to the screen
    sig_11_screen = []
    sig_12_screen = []
    sig_22_screen = []
    
    kLlist = k*l
    for kL in kLlist:
        mat2 = quad_drift_mat2(kL, Lquad=l, Ltot=d)
        sigma1 = propagate_sigma(sigma0, mat2)
        sig_11_screen.append(sigma1[0,0])
        sig_12_screen.append(sigma1[0,1])        
        sig_22_screen.append(sigma1[1,1])        

    sig_11_screen = np.array(sig_11_screen)
    sig_12_screen = np.array(sig_12_screen)
    sig_22_screen = np.array(sig_22_screen)
    
    beta  = sig_11_screen/emit
    alpha = -sig_12_screen/emit
    gamma = sig_22_screen/emit
    
    # Form bmag
    gamma0 = (1+alpha0**2)/beta0
    bmag = (beta * gamma0 - 2*alpha * alpha0 + gamma * beta0) / 2
    bmag = np.min(bmag) # TODO: FIT AND RETURN MIN OF FIT INSTEAD (if not getting at given Q value)
        
    # ignoring correlations
    # TODO: check error propagation for bmag
    bmag_err = bmag * np.sqrt((c2_err/c2)**2 + (c1_err/c1)**2 + (c0_err/c0)**2)
    return bmag, bmag_err

def get_normemit(energy, xrange, yrange, xrms, yrms, xrms_err=None, yrms_err=None,\
                 adapt_ranges=False, num_points=5, show_plots=False):
    """Returns normalized emittance [m]
       given quad values and beamsizes""" 
    if np.isnan(xrms).any() or np.isnan(yrms).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    mkdir_p("plots")
    gamma = energy/m_0
    beta = np.sqrt(1-1/gamma**2)

    kx = get_k1(get_gradient(xrange), beta*energy)
    ky = get_k1(get_gradient(yrange), beta*energy)
    
    emitx, emitx_err, coefsx, coefsx_err, kx_final = fit_sigma(np.array(xrms), kx, axis='x', sizes_err=xrms_err,\
                                       adapt_ranges=adapt_ranges, num_points=num_points, show_plots=show_plots)

    emity, emity_err, coefsy, coefsy_err, ky_final = fit_sigma(np.array(yrms), -ky, axis='y', sizes_err=yrms_err,\
                                       adapt_ranges=adapt_ranges, num_points=num_points, show_plots=show_plots)
    
    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    
    if np.isnan(emitx) or np.isnan(emity):
        save_data(timestamp,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,xrms,yrms,kx,ky,str(adapt_ranges))
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    bmagx, bmagx_err = get_bmag(coefsx, coefsx_err, kx_final, emitx, emitx_err, axis='x')
    bmagy, bmagy_err = get_bmag(coefsy, coefsy_err, ky_final, emity, emity_err, axis='y')    
        
    norm_emitx = emitx*gamma*beta
    norm_emitx_err = emitx_err*gamma*beta
    norm_emity = emity*gamma*beta
    norm_emity_err = emity_err*gamma*beta 
    
    # log data

    save_data(timestamp,norm_emitx,norm_emity,bmagx,bmagy,norm_emitx_err,norm_emity_err,bmagx_err,bmagy_err,\
              str(np.array(xrms)),str(np.array(yrms)),str(kx),str(ky),str(adapt_ranges))
    
    numpy_save(norm_emitx,norm_emity,bmagx,bmagy,norm_emitx_err,norm_emity_err,bmagx_err,bmagy_err, timestamp=timestamp)
    
    
    #print(adapt_ranges)
    print(f"nemitx: {norm_emitx/1e-6:.2f}, nemity: {norm_emity/1e-6:.2f}")
    print(f"nemitx_err: {norm_emitx_err/1e-6:.2f}, nemity_err: {norm_emity_err/1e-6:.2f}")
    print(f"bmagx: {bmagx:.2f}, bmagy: {bmagy:.2f}")
    print(f"bmagx_err: {bmagx_err:.2f}, bmagy_err: {bmagy_err:.2f}")
    

    
    return norm_emitx, norm_emity, bmagx, bmagy, norm_emitx_err, norm_emity_err, bmagx_err, bmagy_err

def plot_fit(x, y, x_fit, axis, yerr=None, save_plot=False, show_plots=False, title_suffix=""):
    """Plot and save the emittance fits of size**2 vs k"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7,5))
    
    # plot x-axis in kG and sizes in um
    sign = -1 if axis=="y" else 1
    x = sign*get_quad_field(x)
    x_fit_gauss = sign*get_quad_field(x_fit)
    
    if yerr is not None and yerr.all() > 0:
        abs_sigma = True
        yerr_plot = np.array(yerr/1e-6) # for plotting
    else: 
        abs_sigma = False
        yerr_plot = None
        
    # fit just for plotting
    coefs, cov = curve_fit(func, x, y, sigma=yerr, absolute_sigma=abs_sigma)
    y_fit = np.array(np.polyval(coefs, x_fit_gauss))

    plt.errorbar(x, y/1e-6, yerr=yerr_plot, marker="x")
    plt.plot(x_fit_gauss, y_fit/1e-6)
        
    plt.xlabel(r"B (kG)")
    plt.ylabel(r"sizes ($\mu$m)")
    plt.title(f"{axis}-axis "+title_suffix)
    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    
    # DEBUGGING
    if save_plot:

        plt.savefig(savepaths['fits'] + f"emittance_{axis}_fit_{timestamp}.png", dpi=100)

    if show_plots:
        plt.show()
    plt.close()
        
def get_quad_field(k, energy=energy, l=l): 
    """Get quad field [kG] from k1 [1/m^2]"""
    gamma = energy/m_0
    beta = np.sqrt(1-1/gamma**2)
    return np.array(k)*l/0.1/0.2998*energy*beta

def adapt_range(x, y, axis, w=None, fit_coefs=None, x_fit=None, energy=energy, num_points=5, save_plot=False, show_plots=True):
    """Returns new scan quad values if called without initial fit coefs"""
    """Returns new coefs if called from fit_sigma with initial fit coefs"""
    if w is None:
        abs_sigma = False
    else: 
        abs_sigma = True

    if fit_coefs is None:
        return_range = True
        
        gamma = energy/m_0
        beta = np.sqrt(1-1/gamma**2)
        k = get_k1(get_gradient(x), beta*energy)        

        if axis == 'x':
            min_k, max_k = np.min(k), 0
        elif axis == 'y':
            k=-k
            min_k, max_k = np.min(k), np.max(k)

        fit_coefs, fit_cov = curve_fit(func, k, y**2, sigma=w, absolute_sigma=abs_sigma)

        x_fit = np.linspace(min_k, max_k, 100)
        x=k
    else:
        return_range = False 
        
    if axis == 'x':
        min_x, max_x = np.min(x), 0
        # quad ranges 0 to -10 kG for scanning
        min_x_range, max_x_range = -20.0, -13.0
    elif axis == 'y':
        min_x, max_x = np.min(x), np.max(x)
        # quad ranges 0 to -10 kG for scanning
        min_x_range, max_x_range = 13, 20.0
        
    c2, c1, c0 = fit_coefs
    
    if c2<0:
        concave_function = True
    else:
        concave_function = False
    
    # find range within 2-3x the focus size 
    y_lim = np.min(np.polyval(fit_coefs, x_fit))*1.3
    if y_lim<0:
        print(f"{axis} axis: min. of poly fit is negative. Setting it to 0.")
        y_lim = np.mean(y**2)/5
    roots = np.roots((c2, c1, c0-y_lim))
    # Flag bad fit with complex roots
    if np.iscomplex(roots).any():
        print("Cannot adapt quad ranges, complex root encountered.")
        raise ComplexRootError
    
    # if roots are outside quad scanning range, set to scan range lim
    if np.min(roots)<min_x_range:
        roots[np.argmin(roots)] = min_x_range
    if np.max(roots)>max_x_range:
        roots[np.argmax(roots)] = max_x_range
    # have at least 3 scanning points within roots
    range_fit = np.max(roots)-np.min(roots)
    if range_fit<2:
        # need at least 3 points for polynomial fit within a range to see beamsize changes
        x_fine_fit = np.linspace(np.min(roots)-1.5, np.max(roots)+1.5, num_points)
        
    if concave_function:
        print("Adjusting concave poly.")
        # go to lower side of concave polynomials 
        # (assuming it is closer to the local minimum)
        x_min_concave = x[np.argmin(y)]
        #find the direction of sampling to minimum
        if (x[np.argmin(y)] - x[np.argmin(y)-2])<0:
            x_max_concave = min_x_range
        else:
            x_max_concave = max_x_range
        if (x_max_concave-x_min_concave)>8:
            # if range is too big (>8 1/m^2), narrow it down on the larger side
            x_min_concave = x_min_concave - 4        #-9
        x_fine_fit = np.linspace(x_min_concave, x_max_concave, num_points)
        
    elif (np.max(roots)-np.min(roots))>8:
        # need to concentrate around min!
        dist_min = np.abs(x[np.argmin(y)]-np.min(roots))
        dist_max = np.abs(x[np.argmin(y)]-np.max(roots))
        if dist_min<dist_max:
            x_fine_fit = np.linspace(np.min(roots), np.max(roots)-4, num_points)
        elif dist_min>dist_max:
            x_fine_fit = np.linspace(np.min(roots)+4, np.max(roots), num_points)
        else:
            x_fine_fit = np.linspace(np.min(roots)+2, np.max(roots)-2, num_points)
            
    else:
        x_fine_fit = np.linspace(roots[0], roots[1], num_points)

    if return_range:
        # if this function is called without initial scan
        # return the new quad measurement range for this axis (in kG!!)
        sign = -1 if axis=="y" else 1
        return np.array([sign*get_quad_field(ele) for ele in x_fine_fit])
        
    # GET NEW BEAMSIZES if returning new coefs to emit fn
    # this takes B in kG not K
    ax_idx_size = 1 if axis=="y" else 0
    ax_idx_err = 3 if axis=="y" else 2
    sign = -1 if axis=="y" else 1
    
    fine_fit_sizes, fine_fit_sizes_err = [], []
    for ele in x_fine_fit:
        setquad(sign*get_quad_field(ele))
        beamsizes = get_sizes()
        #print(beamsizes)
        fine_fit_sizes.append(beamsizes[ax_idx_size])
        fine_fit_sizes_err.append(beamsizes[ax_idx_err])
        #print(fine_fit_sizes)

    if np.isnan(fine_fit_sizes).any():
        not_nan_array = ~np.isnan(fine_fit_sizes)
        fine_fit_sizes = np.array(fine_fit_sizes)[not_nan_array]
        fine_fit_sizes_err = np.array(fine_fit_sizes_err)[not_nan_array]
        x_fine_fit = np.array(x_fine_fit)[not_nan_array]
        if (len(fine_fit_sizes)<3) or (True not in not_nan_array):
            return np.nan, np.nan

    fine_fit_sizes, fine_fit_sizes_err = np.array(fine_fit_sizes), np.array(fine_fit_sizes_err)
    if np.sum(fine_fit_sizes_err)==0:
        w = None
        abs_sigma = False
    else:
        w = 2*fine_fit_sizes*fine_fit_sizes_err # since we are squaring the beamsize
        abs_sigma = True
        
    # fit
    coefs, cov = curve_fit(func, x_fine_fit, fine_fit_sizes**2, sigma=w, absolute_sigma=abs_sigma)
    xfit = np.linspace(np.min(x_fine_fit),np.max(x_fine_fit), 100)
    plot_fit(x_fine_fit, fine_fit_sizes, xfit, yerr=fine_fit_sizes_err, axis=axis,\
             save_plot=save_plot, show_plots=show_plots, title_suffix=" - adapted range")
    return coefs, cov, x_fine_fit


def numpy_save(norm_emitx,norm_emity,bmagx,bmagy,norm_emitx_err,norm_emity_err,bmagx_err,bmagy_err,  timestamp=False,savelist = pv_savelist['scalars'],path =savepaths['emit_saves']):
    
    
    ts = isotime()
    x = epics.caget_many(savelist)
    x.append(ts)
    if timestamp:
        x.append(timestamp)
    else:
        x.append(ts)
        
    x.append(norm_emitx)
    x.append(norm_emity)
    x.append(bmagx)
    x.append(bmagy)
    x.append(norm_emitx_err)
    x.append(norm_emity_err)
    x.append(bmagx_err)
    x.append(bmagy_err)
    
    img=epics.caget('PROF:IN10:571:Image:ArrayData')
    nrow = epics.caget('PROF:IN10:571:Image:ArraySize0_RBV')
    ncol = epics.caget('PROF:IN10:571:Image:ArraySize1_RBV')    

    res = epics.caget('PROF:IN10:571:RESOLUTION')
    
    nrow1 = caget("CAMR:LT10:900:Image:ArraySize0_RBV")
    ncol1 = caget("CAMR:LT10:900:Image:ArraySize1_RBV")

    resolution1 = caget("CAMR:LT10:900:RESOLUTION")
    im1 =caget('CAMR:LT10:900:Image:ArrayData')
    

    np.save(path+ts+'_571_img_.npy',img.reshape((ncol,nrow)))
    np.save(path+ts+'_571_res_.npy',np.array(res))
    
    np.save(path+ts+'_x_.npy',np.array(x))

    np.save(path+ts+'_vcc_img_.npy',im1.reshape((ncol1,nrow1)))
    np.save(path+ts+'_vcc_res_.npy',np.array(resolution1))

def save_data(timestamp, nex, ney, bmx, bmy, nex_err, ney_err, bmx_err, bmy_err, xsizes, ysizes, kx, ky, adapted):
    f= open(savepaths['summaries']+"emit_calc_log.csv", "a+")
    f.write(f"{timestamp},{nex},{ney},{bmx},{bmy},{xsizes},{ysizes},{kx},{ky},{adapted}\n")
    f.close()

class ConcaveFitError(Exception):
    """Raised when the adapted range emit 
    fit results in concave polynomial"""
    pass

class ComplexRootError(Exception):
    """Raised when the adapted range emit 
    fit results in polynomial with complex root(s)"""
    pass
