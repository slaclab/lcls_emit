import numpy.polynomial.polynomial as poly
import numpy as np 
import warnings

## TODO: beamline info needs to go to devices JSON file 

m_0 = 0.511*1e-3 # mass in [GeV]
d = 2.2 # [m] distance between Q525 and OTR2
l = 0.108 # effective length [m]

def getGradient(b_field, l_eff=0.108):
    ''' Returns the quad field gradient [T/m] 
        l_eff: effective length [m] 
        b_field: integrated field [kG] '''
    return np.array(b_field) *0.1 /l_eff
    
def getK1(g, p):
    '''Returns quad strength [1/m^2]
       g: quad field gradient [T/m]
       p: momentum [GeV] (or alternatively beta*E [GeV])'''
    return 0.2998 * g / p

def fitSigma(sizes, k, d=d, l=l):
    '''Fit sizes^2 = c0 + c1*k + c2*k^2
       returns: matrix elements at quadrupole '''
    coefs = poly.polyfit(k, sizes**2, 2)
   
    # matrix elements at quad 525
    c0, c1, c2 = coefs 
    
    sig_11 = c2 / (d*l)**2
    sig_12 = (c1 - 2*d*l*sig_11) / (2*d**2*l)
    sig_22 = (c0 - sig_11 - 2*d*sig_12) / d**2

    ### plotting 
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(7,5))
#     ffit = np.poly1d(coefs)
#     xfit = np.linspace(np.min(k),0,100)
#     plt.plot(k, sizes**2, marker="x")
#     plt.plot(xfit, poly.polyval(xfit, coefs))
    ### end plotting
    
    return sig_11, sig_12, sig_22

def getEmit(sig11, sig12, sig22):
    '''Returns emittance (not normalized)'''
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            print(sig11*sig22 - sig12**2)
            emit  = np.sqrt(sig11*sig22 - sig12**2)
            return emit
    except RuntimeWarning:
        return 1000 # arbitrary high value

def getNormEmit(energy, quad_vals, xrms, yrms):
    '''Returns normalized emittance [m]
       given quad values and beamsizes'''
    gamma = energy/m_0

    b1_gradient = getGradient(quad_vals)
    k = getK1(b1_gradient, energy)

    sig_11, sig_12, sig_22 = fitSigma(xrms, k)
    emitx = getEmit(sig_11, sig_12, sig_22)

    sig_11, sig_12, sig_22 = fitSigma(yrms, k)
    emity = getEmit(sig_11, sig_12, sig_22)
    
    # debugging emity and emitx separately
    #print("nemitx ",emitx*gamma/1e-6," nemity ",emity*gamma/1e-6)

    if emitx == 1000 or emity == 1000:
        return 1000

    return np.sqrt(emitx * emity)*gamma
