import datetime
import numpy as np
import warnings
import sys, os, errno
import scipy
import time
from scipy.optimize import curve_fit

# on sim
# from beam_io_sim import get_beamsizes

# on lcls
# from beam_io import get_beamsizes, setquad, quad_control

import json
from os.path import exists

# import epics
# from epics import caget, caput
# def isotime():
# return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()
# isotime()


# do not display warnings when cov can't be computed
# this will happen when len(y)<=3 and yerr=0
warnings.simplefilter('ignore', scipy.optimize.OptimizeWarning)

# Set rootp directory
dirname = os.path.dirname(os.path.abspath(__file__))
rootp = os.path.join(dirname, '')

beamline_info = json.load(open(rootp + 'config_files/beamline_info.json'))
meas_pv_info = json.load(open(rootp + 'config_files/meas_pv_info.json'))

m_0 = beamline_info['m_0']
d = beamline_info['d']
l = beamline_info['l']
twiss0 = beamline_info['Twiss0']
energy = beamline_info['energy']

# scanning quad range
meas_min = meas_pv_info['meas_device']['pv']['min']
meas_max = meas_pv_info['meas_device']['pv']['max']

# load info about where to put saving of raw images and summaries; make directories if needed and start headings
savepaths = json.load(open(rootp + 'config_files/savepaths.json'))

pv_savelist = json.load(open(rootp + 'config_files/save_scalar_pvs.json'))


def mkdir_p(path):
    """Set up dirs for results in working dir"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# mkdir_p(savepaths['summaries'])

# file_exists = exists(savepaths['summaries']+"emit_calc_log.csv")

# if not file_exists:

##todo add others as inputs
# f= open(savepaths['summaries']+"emit_calc_log.csv", "a+")
# f.write(f"{'timestamp'},{'nex'},{'ney'},{'bmx'},{'bmy'},{'xsizes'},{'ysizes'},{'kx'},{'ky'},{'adapted'}\n")
# f.close()

def func(x, a, b, c):
    """Polynomial function for emittance fit"""
    return a * x ** 2 + b * x + c


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
    return np.array(b_field) * 0.1 / l_eff


def get_k1(g, p):
    """Returns quad strength [1/m^2]
       g: quad field gradient [T/m]
       p: momentum [GeV] (or alternatively beta*E [GeV])"""
    return 0.2998 * g / p


def fit_sigma(sizes, k, axis, sizes_err=None, d=d, l=l, adapt_ranges=False, num_points=5, show_plots=False):
    """Fit sizes^2 = c0 + c1*k + c2*k^2
       returns: c0, c1, c2"""
    sizes = np.array(sizes)
    k = np.array(k)

    # ---------- NOTE: from update on dec 9
    # cutoff = 4.0
    # idx = np.argwhere(sizes < cutoff * np.min(sizes)).flatten()
    # sizes = sizes[idx]
    # k = k[idx]
    # ----------

    if len(sizes) < 3:
        # print("Less than 3 data points were passed.")
        # return np.nan, np.nan, np.nan, np.nan, np.nan
        return np.nan

    if sizes_err is not None and sizes.all() > 0 and sizes_err.all() > 0:
        w = 2 * sizes * np.array(sizes_err)  # sigma for poly fit
        w = np.sqrt(w ** 2 + sizes ** 2)  # adding the sizes as extra weights
        abs_sigma = True
    else:
        w = None
        abs_sigma = False
    coefs, cov = curve_fit(func, k, sizes ** 2, sigma=w, absolute_sigma=abs_sigma)

    if axis == 'x':
        min_k, max_k = np.min(k), np.max(k)
    elif axis == 'y':
        min_k, max_k = np.min(k), np.max(k)

    xfit = np.linspace(min_k, max_k, 100)

    # FOR DEBUGGING ONLY
    plot_fit(k, sizes, xfit, yerr=sizes_err, axis=axis, save_plot=True, show_plots=show_plots)

    # if adapt_ranges:
    # try:
    # coefs, cov, k = adapt_range(k, sizes, w=w, axis=axis, fit_coefs=coefs, x_fit=xfit, energy=energy, num_points=num_points, save_plot=True, show_plots=show_plots)
    ## log data
    # timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    # if axis=="x":
    # save_data(timestamp,0,0,0,0,0,0,0,0,sizes,0,k,0,str(adapt_ranges))
    # if axis=="y":
    # save_data(timestamp,0,0,0,0,0,0,0,0,0,sizes,0,k,str(adapt_ranges))
    ##         except NameError:
    ##             print("Error: A function to get beamsizes is not defined. Returning original fit.")
    ##             plot_fit(k, sizes, xfit, yerr=sizes_err, axis=axis, save_plot=True, show_plots=show_plots)
    # except ComplexRootError:
    # print("Error: Cannot adapt quad ranges. Returning original fit.")
    # plot_fit(k, sizes, xfit, yerr=sizes_err, axis=axis, save_plot=True, show_plots=show_plots)
    # except ConcaveFitError:
    # print("Error: Cannot adapt quad ranges due to concave poly. Returning original fit.")
    # plot_fit(k, sizes, yerr=sizes_err, axis=axis, save_plot=True, show_plots=show_plots)
    # else:
    # plot_fit(k, sizes, xfit, yerr=sizes_err, axis=axis, save_plot=True, show_plots=show_plots)

    if np.isnan(coefs).any() or np.isnan(cov).any():
        print('error coeff:', coef, cov)
        # return np.nan, np.nan, np.nan, np.nan, np.nan
        return np.nan

    # poly.poly: return c0,c1,c2
    # np.polyfit: highest power first
    c2, c1, c0 = coefs
    coefs_err = np.sqrt(np.diag(cov))

    emit2 = (4 * c0 * c2 - c1 ** 2) / l ** 2 / (4 * d ** 4)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            emit = np.sqrt(emit2)
            # error propagation for dependent variables
            emit_gradient = 1./(4*l*d**2*emit) * np.array([[4*c0, -2*c1, 4*c2]]).T
            emit_err = np.sqrt(np.matmul(np.matmul(emit_gradient.T, cov), emit_gradient))
            # return emit, emit_err[0][0], coefs, coefs_err, k
            return emit, emit_err[0][0]
    except RuntimeWarning:
        # print('error:', emit2)
        # return np.nan, np.nan, np.nan, np.nan, np.nan
        return np.nan, np.nan


def propagate_sigma(sigma_mat2, mat2):
    return (mat2 @ sigma_mat2) @ mat2.T


def drift_mat2(L):
    return np.array([[1, L], [0, 1]])


def quad_mat2(kL, L=l):
    """
    Quadrupole transfer matrix, 2x2. Note that

    """

    if L == 0:
        return thin_quad_mat2(kL)

    k = kL / L

    if k == 0:
        mat2 = drift_mat2(L)
    elif k > 0:
        # Focusing
        rk = np.sqrt(k)
        phi = rk * L
        mat2 = [[np.cos(phi), np.sin(phi) / rk], [-rk * np.sin(phi), np.cos(phi)]]
    else:
        # Defocusing
        rk = np.sqrt(-k)
        phi = rk * L
        mat2 = [[np.cosh(phi), np.sinh(phi) / rk], [rk * np.sinh(phi), np.cosh(phi)]]

    return mat2


def quad_drift_mat2(kL, *, Ltot=d + l, Lquad=l):
    """
    Composite [quad, drift] 2x2 transfer matrix.
    """

    Ldrift = Ltot - Lquad

    return drift_mat2(Ldrift) @ quad_mat2(kL, Lquad)


def get_bmag(coefs, coefs_err, k, emit, emit_err, axis, twiss0=twiss0):
    """Calculates Bmag from calculated emittance
    and from initial Twiss at OTR2: HARDCODED from Matlab GUI"""

    c2, c1, c0 = coefs
    c2_err, c1_err, c0_err = coefs_err

    sig11 = c2 / (d * l) ** 2
    sig12 = (-c1 - 2 * d * l * sig11) / (2 * d ** 2 * l)
    sig22 = (c0 - sig11 - 2 * d * sig12) / d ** 2

    # Matrix form
    sigma0 = np.array([[sig11, sig12], [sig12, sig22]])

    beta0 = twiss0[2] if axis == 'x' else twiss0[3] if axis == 'y' else 0
    alpha0 = twiss0[4] if axis == 'x' else twiss0[5] if axis == 'y' else 0
    gamma0 = (1 + alpha0 ** 2) / beta0

    beta_quad = sig11 / emit
    alpha_quad = -sig12 / emit

    # Propagate forward to the screen
    sig_11_screen = []
    sig_12_screen = []
    sig_22_screen = []

    kLlist = k * l
    for kL in kLlist:
        mat2 = quad_drift_mat2(kL, Lquad=l, Ltot=d)
        sigma1 = propagate_sigma(sigma0, mat2)
        sig_11_screen.append(sigma1[0, 0])
        sig_12_screen.append(sigma1[0, 1])
        sig_22_screen.append(sigma1[1, 1])

    sig_11_screen = np.array(sig_11_screen)
    sig_12_screen = np.array(sig_12_screen)
    sig_22_screen = np.array(sig_22_screen)

    # Twiss at screen
    beta = sig_11_screen / emit
    alpha = -sig_12_screen / emit
    gamma = sig_22_screen / emit

    # Form bmag
    gamma0 = (1 + alpha0 ** 2) / beta0
    bmag = (beta * gamma0 - 2 * alpha * alpha0 + gamma * beta0) / 2

    # bmag fn of c2, c1, c0, get bmag err from c2_err, c1_err, c0_err
    # ignoring correlations, TODO: check simpler way to add correlations
    bmag_err = bmag * np.sqrt((c2_err / c2) ** 2 + (c1_err / c1) ** 2 + (c0_err / c0) ** 2)

    # print(
    # f"Min bmag{axis}: {np.min(bmag):.2f} +/- {bmag_err[np.argmin(bmag)]:.2f}")

    # return bmag_list
    return bmag, bmag_err, beta_quad, alpha_quad


# def get_opt_quad(k, bmagx, bmagy, bmagx_err, bmagy_err):
# from numpy.polynomial.polynomial import polyval
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(7, 5))

# bmag = np.sqrt(bmagx * bmagy)
## TODO: is there a better way to do this?
##  (technically a fn of the coeffs, need to take derivatives...)
# bmag_err = bmag * np.sqrt((bmagx_err / bmagx) ** 2 + (bmagy_err / bmagy) ** 2)

# min_idx = np.argmin(bmag)
# bmag_min = np.min(bmag)
# bmag_min_err = bmag_err[min_idx]

## find min of bmag WITHIN uncertainties, x axis in kG
# coefs_bmag_fit, cov_bmag_fit = curve_fit(func, get_quad_field(k), bmag, sigma=bmag_err, absolute_sigma=True)
# c2, c1, c0 = coefs_bmag_fit
# opt_quad_roots = np.roots((c2, c1, c0 - (bmag_min+bmag_min_err)))
# opt_quad_roots = opt_quad_roots
# opt_quad = np.mean(opt_quad_roots)
# opt_quad_err = abs(opt_quad_roots[0] - opt_quad)


## get bmagx/y at min within uncertainties from opt_quad
# """USING FITS"""
## params: polyval( x_val, [c0, c1, c2])
# coefs_x_tmp, cov_x_tmp = curve_fit(func, get_quad_field(k), bmagx, sigma=bmagx_err, absolute_sigma=True)
# coefs_y_tmp, cov_y_tmp = curve_fit(func, get_quad_field(k), bmagy, sigma=bmagy_err, absolute_sigma=True)
# c2x, c1x, c0x = coefs_x_tmp
# c2y, c1y, c0y = coefs_y_tmp
## get bmagx/y at opt_quad
# bmag_x_opt = polyval(opt_quad, [c0x, c1x, c2x])
# bmag_y_opt = polyval(opt_quad, [c0y, c1y, c2y])
## get uncertainty on bmagx/y within opt_quad +/- err
# bmag_x_opt_err_upper = abs(polyval(opt_quad+opt_quad_err, [c0x, c1x, c2x]) - bmag_x_opt)
# bmag_y_opt_err_upper = abs(polyval(opt_quad+opt_quad_err, [c0y, c1y, c2y]) - bmag_y_opt)
# bmag_x_opt_err_lower = abs(polyval(opt_quad-opt_quad_err, [c0x, c1x, c2x]) - bmag_x_opt)
# bmag_y_opt_err_lower = abs(polyval(opt_quad-opt_quad_err, [c0y, c1y, c2y]) - bmag_y_opt)

# opt_vals_from_fits = bmag_x_opt, bmag_x_opt_err_upper, bmag_x_opt_err_lower,\
# bmag_y_opt, bmag_y_opt_err_upper, bmag_y_opt_err_lower

# """USING MEASUREMENTS"""
# bmag_x_opt_meas = bmagx[min_idx]
# bmag_y_opt_meas = bmagy[min_idx]
# bmag_x_opt_err_meas = bmagx_err[min_idx]  # errors are assumed symmetrical
# bmag_y_opt_err_meas = bmagy_err[min_idx]

# opt_vals_from_meas = bmag_x_opt_meas, bmag_x_opt_err_meas, bmag_x_opt_err_meas,\
# bmag_y_opt_meas, bmag_y_opt_err_meas, bmag_y_opt_err_meas

# """PLOTTING FIT METHOD"""
# x = np.linspace(meas_min, meas_max, 20)
# x_fit = np.linspace(meas_min, meas_max, 100)
# plt.errorbar(x, bmag, bmag_err, color="C0", label="bmag")
# plt.errorbar(x, bmagx, bmagx_err, color="C1", label="bmagx")
# plt.errorbar(x, bmagy, bmagy_err, color="C2", label="bmagy")

# plt.errorbar(x_fit, polyval(x_fit, [c0, c1, c2]), color="C0", linestyle="dotted")
# plt.errorbar(x_fit, polyval(x_fit, [c0x, c1x, c2x]),  color="C1", linestyle="dotted")
# plt.errorbar(x_fit, polyval(x_fit, [c0y, c1y, c2y]),  color="C2", linestyle="dotted")

# plt.ylim(0, np.max(bmag))
# plt.xlim(meas_min, meas_max)
# plt.vlines(opt_quad, 0, np.max(bmag), color="gray", linestyle="dashed", alpha=0.5)

## these can plot the other method as lines
## plt.hlines(bmag_x_opt_meas, meas_min, meas_max, color="gray", linestyle="solid", alpha=0.5)
## plt.hlines(bmag_x_opt_meas+bmag_x_opt_err_meas, meas_min, meas_max, color="gray", linestyle="dashed", alpha=0.5)
## plt.hlines(bmag_x_opt_meas-bmag_x_opt_err_meas, meas_min, meas_max, color="gray", linestyle="dashed", alpha=0.5)

## plt.hlines(bmag_x_opt, meas_min, meas_max, color="blue", linestyle="solid", alpha=0.5)
## plt.hlines(bmag_x_opt+bmag_x_opt_err_upper, meas_min, meas_max, color="blue", linestyle="dashed", alpha=0.7)
## plt.hlines(bmag_x_opt-bmag_x_opt_err_lower, meas_min, meas_max, color="blue", linestyle="dashed", alpha=0.7)

# plt.ylabel(r"Bmag (geometric mean)")
# plt.xlabel(r"B (kG)")
# plt.title("Bmag at OTR2 vs Q525 strength")
# plt.legend()
# timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")

# save_plot = True
## DEBUGGING
# if save_plot:
# plt.savefig(savepaths['fits'] + f"bmag_otr2_{timestamp}.png", dpi=100)
# plt.show()
# plt.close()

# return bmag_min, bmag_min_err, opt_quad, opt_quad_err, opt_vals_from_fits, opt_vals_from_meas

def get_normemit(energy, xrange, yrange, xrms, yrms, xrms_err=None, yrms_err=None,
                 adapt_ranges=False, num_points=5, show_plots=False):
    """Returns normalized emittance [m]
       given quad values and beamsizes"""
    if np.isnan(xrms).any() or np.isnan(yrms).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # mkdir_p("plots")
    gamma = energy / m_0
    beta = np.sqrt(1 - 1 / gamma ** 2)

    # get init quad value in kGauss
    # init_quad = quad_control(action="get")

    kx = get_k1(get_gradient(xrange), beta * energy)
    ky = get_k1(get_gradient(yrange), beta * energy)

    # emitx, emitx_err, coefsx, coefsx_err, kx_final = fit_sigma(np.array(xrms), kx, axis='x', sizes_err=xrms_err,
    emitx, emitx_err = fit_sigma(np.array(xrms), kx, axis='x', sizes_err=xrms_err,
                      adapt_ranges=adapt_ranges, num_points=num_points, show_plots=show_plots)

    # emity, emity_err, coefsy, coefsy_err, ky_final = fit_sigma(np.array(yrms), -ky, axis='y', sizes_err=yrms_err,
    emity, emity_err = fit_sigma(np.array(yrms), -ky, axis='y', sizes_err=yrms_err,
                      adapt_ranges=adapt_ranges, num_points=num_points, show_plots=show_plots)

    # timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")

    if np.isnan(emitx) or np.isnan(emity):
        # print(emitx, emity)
        # save_data(timestamp,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,xrms,yrms,kx,ky,str(adapt_ranges))
        # return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return np.nan, np.nan

    ## return quad to init value TODO: do this at every return statement
    ##quad_control(init_quad, action="set")

    ## taking full quad range from 0 to -10 kG
    # k = np.linspace(get_k1(get_gradient(meas_min), beta*energy), get_k1(get_gradient(meas_max), beta*energy), 20)

    # bmagx, bmagx_err, beta_quad_x, alpha_quad_x = get_bmag(coefsx, coefsx_err, k, emitx, emitx_err,
    # axis='x')
    # bmagy, bmagy_err, beta_quad_y, alpha_quad_y = get_bmag(coefsy, coefsy_err, -k, emity, emity_err,
    # axis='y')

    # bmag_min, bmag_min_err, opt_quad, opt_quad_err, opt_vals_from_fits, opt_vals_from_meas = get_opt_quad(k,
    # bmagx,
    # bmagy,
    # bmagx_err,
    # bmagy_err)

    ## unpack bmagx/y results
    # bmagx_from_fit, bmagx_from_fit_err_upper, bmagx_from_fit_err_lower = opt_vals_from_fits[0:3]
    # bmagy_from_fit, bmagy_from_fit_err_upper, bmagy_from_fit_err_lower = opt_vals_from_fits[3:]

    # bmagx_from_meas, bmagx_from_meas_err_upper, bmagx_from_meas_err_lower = opt_vals_from_meas[0:3]
    # bmagy_from_meas, bmagy_from_meas_err_upper, bmagy_from_meas_err_lower = opt_vals_from_meas[3:]

    # """Choose which to use/save"""
    ## TODO: taking mean of errors here, maybe another way is better
    # bmagx = bmagx_from_fit
    # bmagx_err = np.mean(bmagx_from_fit_err_upper, bmagx_from_fit_err_lower)
    # bmagy = bmagy_from_fit
    # bmagy_err = np.mean(bmagy_from_fit_err_upper, bmagy_from_fit_err_lower)

    norm_emitx = emitx * gamma * beta
    norm_emitx_err = emitx_err * gamma * beta
    norm_emity = emity * gamma * beta
    norm_emity_err = emity_err * gamma * beta

    ##hardcoded
    # epics.caput('QUAD:IN10:525:BCTRL',opt_quad)
    ## log data
    # save_data(timestamp,norm_emitx,norm_emity,bmagx,bmagy,norm_emitx_err,norm_emity_err,bmagx_err,bmagy_err,
    # str(np.array(xrms)),str(np.array(yrms)),str(kx),str(ky),str(adapt_ranges))
    # numpy_save(norm_emitx,norm_emity,bmagx,bmagy,norm_emitx_err,norm_emity_err,bmagx_err,bmagy_err,
    # beta_quad_x,alpha_quad_x,beta_quad_y,alpha_quad_y,bmag_min,opt_quad,timestamp=timestamp)

    # print(fr"nemitx: {norm_emitx / 1e-6:.2f} +/- {norm_emitx_err / 1e-6:.2f} mm mrad")
    # print(f"nemity: {norm_emity / 1e-6:.2f} +/- {norm_emity_err / 1e-6:.2f} mm mrad")
    # print(f"Bmag at minimum: {bmag_min:.2f} +/- {bmag_min_err:.2f}")
    # print(f"Optimal quad: {opt_quad:.2f} +/- {abs(opt_quad_err):.2f} kG")
    # print("=== from poly fits ===")
    # print(f"Bmagx at opt quad: {bmagx_from_fit:.2f} + {bmagx_from_fit_err_upper:.2f} - {bmagx_from_fit_err_lower:.2f}")
    # print(f"Bmagy at opt quad: {bmagy_from_fit:.2f} + {bmagy_from_fit_err_upper:.2f} - {bmagy_from_fit_err_lower:.2f}")
    # print("=== from measurements ===")
    # print(f"Bmagx at opt quad: {bmagx_from_meas:.2f} +/- {bmagx_from_meas_err_upper:.2f}")
    # print(f"Bmagy at opt quad: {bmagy_from_meas:.2f} +/- {bmagy_from_meas_err_upper:.2f}")

    # return norm_emitx, norm_emity, bmagx, bmagy, norm_emitx_err, norm_emity_err, bmagx_err, bmagy_err, bmag_min, opt_quad
    return norm_emitx, norm_emity, norm_emitx_err, norm_emity_err


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
        yerr_plot = np.array(yerr)/1e-6 # for plotting
    else:
        abs_sigma = False
        yerr_plot = None

    # fit just for plotting in um
    coefs, cov = curve_fit(func, x, y/1e-6, sigma=yerr_plot, absolute_sigma=abs_sigma, method='trf')
    y_fit = np.array(np.polyval(coefs, x_fit_gauss)) # in um

    plt.errorbar(x, y/1e-6, yerr=yerr_plot, marker="x")
    plt.plot(x_fit_gauss, y_fit)

    plt.xlabel(r"B (kG)")
    plt.ylabel(r"sizes ($\mu$m)")
    plt.title(f"{axis}-axis "+title_suffix)
    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")

    # # DEBUGGING
    #if save_plot:
        #plt.savefig(savepaths['fits'] + f"emittance_{axis}_fit_{timestamp}.png", dpi=100)

    if show_plots:
        plt.show()
    plt.close()

def get_quad_field(k, energy=energy, l=l):
    """Get quad field [kG] from k1 [1/m^2]"""
    gamma = energy / m_0
    beta = np.sqrt(1 - 1 / gamma ** 2)
    return np.array(k) * l / 0.1 / 0.2998 * energy * beta


def check_symmetry(rms, rms_err, quad, axis):
    if len(rms) != len(quad):
        raise Exception('Array lengths do not match!')

    left_side = np.argmin(rms)
    right_side = len(quad) - left_side - 1
    stepsize = abs((quad[0] - quad[-1]) / len(quad))

    if left_side == right_side:
        return None

    elif left_side > right_side:
        add = "right"
        diff = left_side - right_side
        # add points to right_side
        xmin = quad[-1] + stepsize
        xmax = xmin + diff * stepsize
        xadd = np.linspace(xmin, xmax, diff)

    elif right_side > left_side:
        add = "left"
        diff = right_side - left_side
        # add points to left_side
        xmin = quad[0] - diff * stepsize
        xmax = quad[0] - stepsize
        xadd = np.linspace(xmin, xmax, diff)

    # get beamsizes for xadd
    # this takes B in kG not K
    ax_idx_size = 1 if axis == "y" else 0
    ax_idx_err = 3 if axis == "y" else 2
    sign = -1 if axis == "y" else 1

    rms_add, rms_err_add = [], []
    for ele in xadd:
        setquad(sign * get_quad_field(ele))
        time.sleep(3)
        beamsizes = get_beamsizes()
        rms_add.append(beamsizes[ax_idx_size])
        rms_err_add.append(beamsizes[ax_idx_err])

    # then append to rms and quad
    if add == "left":
        new_quad_list = list(xadd) + list(quad)
        new_rms_list = list(rms_add) + list(rms)
        new_rms_err_list = list(rms_err_add) + list(rms_err)
    else:
        new_quad_list = list(quad) + list(xadd)
        new_rms_list = list(rms) + list(rms_add)
        new_rms_err_list = list(rms_err) + list(rms_err_add)

    return np.array(new_quad_list), np.array(new_rms_list), np.array(new_rms_err_list)

def adapt_range(x, y, axis, w=None, fit_coefs=None, x_fit=None, energy=energy, num_points=5, save_plot=False,
                show_plots=False):
    """Returns new scan quad values if called without initial fit coefs"""
    """Returns new coefs if called from fit_sigma with initial fit coefs"""
    x = np.array(x)
    y = np.array(y)

    # ---------- NOTE: from update on dec 9
    # cutoff = 4.0
    # idx = np.argwhere(y < cutoff * np.min(y)).flatten()
    # x = x[idx]
    # y = y[idx]
    # # ----------

    if w is None:
        abs_sigma = False
    else:
        abs_sigma = True

    if fit_coefs is None:
        return_range = True

        gamma = energy / m_0
        beta = np.sqrt(1 - 1 / gamma ** 2)
        k = get_k1(get_gradient(x), beta * energy)

        if axis == 'x':
            min_k, max_k = np.min(k), 0
        elif axis == 'y':
            k = -k
            min_k, max_k = np.min(k), np.max(k)

        if w is not None:
            w = np.sqrt(w ** 2 + y ** 2)  # adding the sizes as extra weights
        fit_coefs, fit_cov = curve_fit(func, k, y ** 2, sigma=w, absolute_sigma=abs_sigma, method='trf')

        x_fit = np.linspace(min_k, max_k, 100)

        x = k

        plot_fit(k, y, x_fit, yerr=None, axis=axis,
                 save_plot=False, show_plots=show_plots, title_suffix="init")

    else:
        return_range = False

    min_x_range, max_x_range = x[np.argmin(y)] - 2.5, x[np.argmin(y)] + 2.5
    # if axis == 'x':
    #     min_x, max_x = np.min(x), 0
    #     # quad ranges 0 to -10 kG for scanning
    #     # min_x_range, max_x_range = -16.0,-8.9#-20.0, -13.0
    #     min_x_range, max_x_range =   x[np.argmin(y)]-2.5, 0 # x[np.argmin(y)]+3  #np.min(x), 0 #np.max(x)   #HERE + what to do on machine
    # elif axis == 'y':
    #     min_x, max_x = np.min(x), np.max(x)
    #     # quad ranges 0 to -10 kG for scanning
    #     # min_x_range, max_x_range = 8.9, 16.0
    #     min_x_range, max_x_range =  x[np.argmin(y)]-2.5, x[np.argmin(y)]+2.5  #HERE these should be larger

    c2, c1, c0 = fit_coefs
    if c2 < 0:  # constraining for posterior function samples
        c2 = 1e-10

    if c2 < 0:
        concave_function = True
    else:
        concave_function = False

    # find range within 2-3x the focus size
    # cutoff = 1.2-1.3 for lcls
    # cutoff = 4 for facet and surrogate
    cutoff = 1.5
    y_min_poly = np.min(np.polyval(fit_coefs, x_fit))
    y_lim = np.min(y ** 2) * cutoff

    if y_lim<y_min_poly:
        # in this case the roots won't exist
        y_lim = y_min_poly * cutoff

    if y_lim < 0:
        print(f"{axis} axis: min. of poly fit is negative.")
        y_lim = np.mean(y ** 2) / 5

    # if y_lim < c0:  # constraining for posterior function samples
    # only use with bax
    #     y_lim = c0 * (5 / 4)

    roots = np.roots([c2, c1, c0 - y_lim])

    # Flag bad fit with complex roots
    if np.iscomplex(roots).any():
        print("Cannot adapt quad ranges, complex root encountered.")
        raise ComplexRootError

    # if roots are outside quad scanning range, set to scan range lim
    if np.min(roots) < min_x_range:
        roots[np.argmin(roots)] = min_x_range
    if np.max(roots) > max_x_range:
        roots[np.argmax(roots)] = max_x_range
    # have at least 3 scanning points within roots
    range_fit = np.max(roots) - np.min(roots)
    #     if range_fit<2:
    #         # need at least 3 points for polynomial fit within a range to see beamsize changes
    #         x_fine_fit = np.linspace(np.min(roots)-1.5, np.max(roots)+1.5, num_points)

    if concave_function:
        print("Adjusting concave poly.")
        # go to lower side of concave polynomials
        # (assuming it is closer to the local minimum)
        x_min_concave = x[np.argmin(y)]
        # find the direction of sampling to minimum
        if (x[np.argmin(y)] - x[np.argmin(y) - 2]) < 0:
            x_max_concave = min_x_range
        else:
            x_max_concave = max_x_range
        if (x_max_concave - x_min_concave) > (max_x_range - min_x_range):
            # if range is too big (in 1/m^2), narrow it down on the larger side
            # print("Range too large, setting to max quad range.")
            x_min_concave = min_x_range
            x_max_concave = max_x_range
        x_fine_fit = np.linspace(x_min_concave, x_max_concave, num_points)

    elif (np.max(roots) - np.min(roots)) > (max_x_range - min_x_range):
        #         # need to concentrate around min!
        #         dist_min = np.abs(x[np.argmin(y)]-np.min(roots))
        #         dist_max = np.abs(x[np.argmin(y)]-np.max(roots))
        #         if dist_min<dist_max:
        #             diff = dist_max-dist_min
        #             x_fine_fit = np.linspace(np.min(roots), np.max(roots)-diff, num_points)
        #         elif dist_min>dist_max:
        #             diff = dist_min-dist_max
        #             x_fine_fit = np.linspace(np.min(roots)+diff, np.max(roots), num_points)
        #         else:
        #             x_fine_fit = np.linspace(np.min(roots)+2, np.max(roots)-2, num_points)
        # print("Range too large, setting to max quad range.")
        x_fine_fit = np.linspace(min_x_range, max_x_range, num_points)

    else:
        x_fine_fit = np.linspace(roots[0], roots[1], num_points)

    # if return_range:
    # if this function is called without initial scan
    # return the new quad measurement range for this axis (in kG!!)
    vsign = -1 if axis == "y" else 1
    ret_list = [vsign * get_quad_field(ele) for ele in x_fine_fit]
    ret = np.array(ret_list)
    return ret

    ## GET NEW BEAMSIZES if returning new coefs to emit fn
    ## this takes B in kG not K
    # ax_idx_size = 1 if axis=="y" else 0
    # ax_idx_err = 3 if axis=="y" else 2
    # sign = -1 if axis=="y" else 1

    # fine_fit_sizes, fine_fit_sizes_err = [], []
    # for ele in x_fine_fit:
    # setquad(sign*get_quad_field(ele))
    # time.sleep(3.0)
    # beamsizes = get_beamsizes()
    ##print(beamsizes)
    # fine_fit_sizes.append(beamsizes[ax_idx_size])
    # fine_fit_sizes_err.append(beamsizes[ax_idx_err])
    ##print(fine_fit_sizes)

    # if np.isnan(fine_fit_sizes).any():
    # not_nan_array = ~np.isnan(fine_fit_sizes)
    # fine_fit_sizes = np.array(fine_fit_sizes)[not_nan_array]
    # fine_fit_sizes_err = np.array(fine_fit_sizes_err)[not_nan_array]
    # x_fine_fit = np.array(x_fine_fit)[not_nan_array]
    # if (len(fine_fit_sizes) < 3) or (True not in not_nan_array):
    # return np.nan, np.nan

    ## check symmetry
    # x_fine_fit, fine_fit_sizes, fine_fit_sizes_err = check_symmetry(fine_fit_sizes, fine_fit_sizes_err, x_fine_fit,
    # axis)

    ## check for NaNs again
    # if np.isnan(fine_fit_sizes).any():
    # not_nan_array = ~np.isnan(fine_fit_sizes)
    # fine_fit_sizes = np.array(fine_fit_sizes)[not_nan_array]
    # fine_fit_sizes_err = np.array(fine_fit_sizes_err)[not_nan_array]
    # x_fine_fit = np.array(x_fine_fit)[not_nan_array]
    # if (len(fine_fit_sizes) < 3) or (True not in not_nan_array):
    # return np.nan, np.nan

    ## TODO: avoid rewriting these! need refactoring
    # if np.isnan(fine_fit_sizes).any():
    # not_nan_array = ~np.isnan(fine_fit_sizes)
    # fine_fit_sizes = np.array(fine_fit_sizes)[not_nan_array]
    # fine_fit_sizes_err = np.array(fine_fit_sizes_err)[not_nan_array]
    # x_fine_fit = np.array(x_fine_fit)[not_nan_array]
    # if (len(fine_fit_sizes) < 3) or (True not in not_nan_array):
    # return np.nan, np.nan

    # fine_fit_sizes, fine_fit_sizes_err = np.array(fine_fit_sizes), np.array(fine_fit_sizes_err)
    # if np.sum(fine_fit_sizes_err)==0:
    # w = None
    # abs_sigma = False
    # else:
    # w = 2*fine_fit_sizes*fine_fit_sizes_err # since we are squaring the beamsize
    # w = np.sqrt(w ** 2 + fine_fit_sizes ** 2)  # adding the sizes as extra weights
    # abs_sigma = True

    ## fit
    # coefs, cov = curve_fit(func, x_fine_fit, fine_fit_sizes**2, sigma=w, absolute_sigma=abs_sigma)
    # xfit = np.linspace(np.min(x_fine_fit),np.max(x_fine_fit), 100)
    # plot_fit(x_fine_fit, fine_fit_sizes, xfit, yerr=fine_fit_sizes_err, axis=axis,\
    # save_plot=save_plot, show_plots=show_plots, title_suffix=" - adapted range")
    # return coefs, cov, x_fine_fit


# def numpy_save(norm_emitx,norm_emity,bmagx,bmagy,norm_emitx_err,norm_emity_err,bmagx_err,bmagy_err,beta_quad_x,alpha_quad_x,beta_quad_y,alpha_quad_y,bmag,opt_quad,timestamp=False,savelist = pv_savelist['scalars'],path =savepaths['emit_saves']):

# ts = isotime()
# x = epics.caget_many(savelist)
# x.append(ts)
# if timestamp:
# x.append(timestamp)
# else:
# x.append(ts)

# x.append(norm_emitx)
# x.append(norm_emity)
# x.append(bmagx)
# x.append(bmagy)
# x.append(norm_emitx_err)
# x.append(norm_emity_err)
# x.append(bmagx_err)
# x.append(bmagy_err)
# x.append(beta_quad_x)
# x.append(alpha_quad_x)
# x.append(beta_quad_y)
# x.append(alpha_quad_y)
# x.append(bmag)
# x.append(opt_quad)


# np.save(path+ts+'_x_.npy',np.array(x))


# def save_data(timestamp, nex, ney, bmx, bmy, nex_err, ney_err, bmx_err, bmy_err, xsizes, ysizes, kx, ky, adapted):
# f= open(savepaths['summaries']+"emit_calc_log.csv", "a+")
# f.write(f"{timestamp},{nex},{ney},{bmx},{bmy},{xsizes},{ysizes},{kx},{ky},{adapted}\n")
# f.close()

class ConcaveFitError(Exception):
    """Raised when the adapted range emit
    fit results in concave polynomial"""
    pass


class ComplexRootError(Exception):
    """Raised when the adapted range emit
    fit results in polynomial with complex root(s)"""
    pass