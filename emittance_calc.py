import numpy.polynomial.polynomial as poly
import datetime
import numpy as np 
import warnings

from beam_io import get_updated_beamsizes

m_0 = 0.511*1e-3 # mass in [GeV]
d = 2.26 # [m] distance between Q525 and OTR2
l = 0.108 # effective length [m]

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

def fit_sigma(sizes, k, axis, d=d, l=l, adapt_ranges=False, num_points=5, show_plots=False):
    """Fit sizes^2 = c0 + c1*k + c2*k^2
       returns: c0, c1, c2"""
    coefs = poly.polyfit(k, sizes**2, 2)
    
    if axis == 'x':
        min_k, max_k = np.min(k), 0
    elif axis == 'y':
        min_k, max_k = np.min(k), np.max(k)
    
    xfit = np.linspace(min_k, max_k, 100)

    #plot_fit(k, sizes, coefs, xfit, axis=axis, save_plot=True, show_plots=show_plots)
    
    if adapt_ranges:
        try:
            coefs = adapt_range(k, sizes, coefs, xfit, axis=axis, num_points=num_points,\
                                save_plot=True, show_plots=show_plots)
            # log data
            timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
            if axis=="x":
                save_data(timestamp,0,0,0,0,sizes,0,k,0,str(adapt_ranges))
            if axis=="y":
                save_data(timestamp,0,0,0,0,0,sizes,0,k,str(adapt_ranges))
#         except NameError:
#             print("Error: A function to get beamsizes is not defined. Returning original fit.")
#             plot_fit(k, sizes, coefs, xfit, axis=axis, save_plot=True, show_plots=show_plots)
        except ValueError:
            print("Error: Cannot adapt quad ranges. Returning original fit.")
            plot_fit(k, sizes, coefs, xfit, axis=axis, save_plot=True, show_plots=show_plots)
        except InvertedFitError:
            print("Error: Cannot adapt quad ranges due to inverted poly. Returning original fit.")
            plot_fit(k, sizes, coefs, xfit, axis=axis, save_plot=True, show_plots=show_plots)     
    else:
        plot_fit(k, sizes, coefs, xfit, axis=axis, save_plot=True, show_plots=show_plots)
        
    # return c0,c1,c2
    c0, c1, c2 = coefs
    
    # matrix elements at quad 525
    sig_11 = c2 / (d*l)**2
    sig_12 = (-c1 - 2*d*l*sig_11) / (2*d**2*l)
    sig_22 = (c0 - sig_11 - 2*d*sig_12) / d**2
    
    return sig_11, sig_12, sig_22

def get_emit(sig11, sig12, sig22):
    """Returns emittance (not normalized)"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            emit  = np.sqrt(sig11*sig22 - sig12**2)
            return emit
    except RuntimeWarning:
        return np.nan 
    
def get_bmag(sig11, sig12, sig22, emit, axis):
    """Calculates Bmag from calculated emittance
    and from initial Twiss at OTR2: HARDCODED from Matlab GUI"""
    # HARDCODED INIT TWISS PARAMS
    twiss0 = [1e-6, 1e-6, 1.113081026, 1.113021659, -6.89403587e-2, -7.029489754e-2]
    
    beta0 =  twiss0[2] if axis == 'x' else twiss0[3] if axis == 'y' else 0
    alpha0 = twiss0[4] if axis == 'x' else twiss0[5] if axis == 'y' else 0
    gamma0 = (1+alpha0**2)/beta0

    beta = sig11/emit
    alpha = -sig12/emit
    gamma = sig22/emit
    
    bmag = 0.5 * (beta*gamma0 - 2*alpha*alpha0 + gamma*beta0)
    return bmag

def get_normemit(energy, xrange, yrange, xrms, yrms, adapt_ranges=False, num_points=5, show_plots=False):
    """Returns normalized emittance [m]
       given quad values and beamsizes"""    
    gamma = energy/m_0
    beta = np.sqrt(1-1/gamma**2)

    kx = get_k1(get_gradient(xrange), beta*energy)
    ky = get_k1(get_gradient(yrange), beta*energy)
    
    sig_11, sig_12, sig_22 = fit_sigma(np.array(xrms), kx, axis='x',\
                                       adapt_ranges=adapt_ranges, num_points=num_points, show_plots=show_plots)
    emitx = get_emit(sig_11, sig_12, sig_22)
    bmagx = get_bmag(sig_11, sig_12, sig_22, emitx, axis='x')

    sig_11, sig_12, sig_22 = fit_sigma(np.array(yrms), -ky, axis='y',\
                                       adapt_ranges=adapt_ranges, num_points=num_points, show_plots=show_plots)
    emity = get_emit(sig_11, sig_12, sig_22)
    bmagy = get_bmag(sig_11, sig_12, sig_22, emity, axis='y')
    
    # log data
    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    save_data(timestamp,emitx*gamma*beta,emity*gamma*beta,bmagx,bmagy,xrms,yrms,kx,ky,str(adapt_ranges))
        
    if np.isnan(emitx) or np.isnan(emity):
        return np.nan, np.nan, np.nan, np.nan
    
    print(f"nemitx: {emitx*gamma*beta/1e-6:.2f}, nemity: {emity*gamma*beta/1e-6:.2f}")
    print(f"bmagx: {bmagx:.2f}, bmagy: {bmagy:.2f}")

    return emitx*gamma*beta, emity*gamma*beta, bmagx, bmagy

def plot_fit(x, y, fit_coefs, x_fit, axis, save_plot=True, show_plots=False, title_suffix=""):
    """Plot and save the emittance fits of size**2 vs k"""
    import matplotlib.pyplot as plt
    import datetime
    fig = plt.figure(figsize=(7,5))
    ffit = np.poly1d(fit_coefs)
    plt.plot(x, y**2, marker="x")
    plt.plot(x_fit, poly.polyval(x_fit, fit_coefs))
    
    plt.xlabel(r"k (1/m$^2$)")
    plt.ylabel(r"sizes$^2$ (m$^2$)")
    plt.title(f"{axis}-axis "+title_suffix)
    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    
    if save_plot:
        try:
            plt.savefig(f"./plots/emittance_{axis}_fit_{timestamp}.png", dpi=100)
        except:
            plt.savefig(f"./emittance_fit_{axis}_{timestamp}.png", dpi=100)
    if show_plots:
        plt.show()
    plt.close()
        
def get_quad_field(k, energy=0.135, l=0.108): 
    """Get quad field [kG] from k1 [1/m^2]"""
    gamma = energy/m_0
    beta = np.sqrt(1-1/gamma**2)
    return np.array(k)*l/0.1/0.2998*energy*beta


def adapt_range(x, y, fit_coefs, x_fit, axis, num_points, save_plot=False, show_plots=True):
    """Adjust scanning range for a given axis to fit around minimum"""
    if axis == 'x':
        min_x, max_x = np.min(x), 0
        # quad ranges 0 to -10 kG for scanning
        min_x_range, max_x_range = -22.2, 0
    elif axis == 'y':
        min_x, max_x = np.min(x), np.max(x)
        # quad ranges 0 to -10 kG for scanning
        min_x_range, max_x_range = 0, 22.2
        
    c0, c1, c2 = fit_coefs
    
    if c2<0:
        inverted_function = True
    else:
        inverted_function = False
    
    # find range within 2-3x the focus size 
    y_lim = np.min(poly.polyval(x_fit, fit_coefs))*4 
    if y_lim<0:
        print(f"{axis} axis: min. of poly fit is negative. Setting it to 0.")
        y_lim = 0
    roots = poly.polyroots((c0-y_lim, c1, c2))
    # Flag bad fit with complex roots
    if np.iscomplex(roots).any():
        print("Bad emittance fit, complex root encountered.")
        raise ValueError("Cannot adapt quad ranges.")
    
    # if roots are outside quad scanning range, set to scan range lim
    if roots[0]<min_x_range:
        roots[0] = min_x_range
    if roots[1]>max_x_range:
        roots[1] = max_x_range
        
    # have at least 3 scanning points within roots
    range_fit = np.max(roots)-np.min(roots)
    if range_fit<2:
        # need at least 3 points for polynomial fit
        x_fine_fit = np.linspace(np.min(roots)-1.5, np.max(roots)+1.5, num_points)
        
    if inverted_function:
        print("Adjusting inverted poly.")
        # go to minimum side of inverted polynomials
        x_min_inverted = x[np.argmin(y)]
        
        #find the direction of sampling to minimum
        if (x[np.argmin(y)] - x[np.argmin(y)-2])<0:
            x_max_inverted = min_x_range
        else:
            x_max_inverted = max_x_range
        
        if (x_max_inverted-x_min_inverted)>8:
            # if range is too big, narrow it down on the larger side
            x_min_inverted = x_min_inverted - 4
            
        x_fine_fit = np.linspace(x_min_inverted, x_max_inverted, num_points)
        
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

        
    # GET NEW BEAMSIZES 
    # this takes B in kG not K
    if axis=="x":
        fine_fit_sizes = np.array([get_updated_beamsizes(get_quad_field(ele))[0] for ele in x_fine_fit])
    elif axis == "y":
        fine_fit_sizes = np.array([get_updated_beamsizes(-get_quad_field(ele))[1] for ele in x_fine_fit])
    
    # fit
    coefs = poly.polyfit(x_fine_fit, fine_fit_sizes**2, 2)
    xfit = np.linspace(np.min([min_x,np.min(x_fine_fit)]),np.max([max_x,np.max(x_fine_fit)]), 100)
    plot_fit(x_fine_fit, fine_fit_sizes, coefs, xfit, axis=axis, save_plot=save_plot, show_plots=show_plots)
    
    return coefs

def save_data(timestamp, nex, ney, bmx, bmy, xsizes, ysizes, kx, ky, adapted):
    f= open(f"emit_calc_log.csv", "a+")
    f.write(f"{timestamp},{nex},{ney},{bmx},{bmy},{xsizes},{ysizes},{kx},{ky},{adapted}\n")
    f.close()
    
class InvertedFitError(Exception):
    """Raised when the adapted range emit 
    fit results in inverted polynomial"""
    pass