import numpy as np
import sys

sys.path.append('/home/fphysics/cemma/git_work/lcls_emit/Emit_from_lcls/tofacet/')

try:
    from epics import caget, caput, PV
except:
    print("did not import epics")
    
from emittance_calc import *
from beam_io import *

# get PV info
pv_info = json.load(open('/home/fphysics/cemma/git_work/lcls_emit/Emit_from_lcls/tofacet/pv_info.json'))

im_pv = pv_info['device']['OTR2']['image']
n_col_pv = pv_info['device']['OTR2']['ncol']
n_row_pv = pv_info['device']['OTR2']['nrow']

meas_input = pv_info['device']['QUAD']['Q525']
varx_pv = pv_info['device']['SOL']['SOL121']
vary_pv = pv_info['device']['QUAD']['Q121']
varz_pv = pv_info['device']['QUAD']['Q122']

energy = caget(pv_info['energy']['DL1'])
resolution = caget(pv_info['device']['OTR2']['resolution'])*1e-6

energy = 0.125
####### Run quad scan ###########################
# example init quad list
#quad_list = [ -6.66, -6.45, -6.31, -6.1667, -6.02, -5.8778, -5.7333, -5.5889, -5.444,-5.3]
quad_list = [ -9.0, -8.5, -8.0, -7.5, -7.0, -6.5]
#quad_list = [ -7.3, -7.05, -6.8, -6.55, -6.3, -5.8, -5.55]
xrms = []
yrms = []
xrms_err = []
yrms_err = []

for quad in quad_list:
    setquad(quad)
    time.sleep(3)
    beamsize = get_beamsizes(use_profMon=False) 
    xrms.append(beamsize[0])
    yrms.append(beamsize[1])
    xrms_err.append(beamsize[2])
    yrms_err.append(beamsize[3])
print(xrms)
print(yrms)

emittance = get_normemit(energy, quad_list, quad_list, np.array(xrms), np.array(yrms), np.array(xrms_err), np.array(yrms_err), adapt_ranges=True, num_points=len(quad_list), show_plots=True)

#################################################

###### Example Quad Scan Data ############################################
#quad_list = [-5.38210006, -4.24154984, -3.71824052, -3.17573903, -1.99949716] 
#xrms = np.array([1.59568569e-04, 8.75908038e-05, 6.55396626e-05, 4.85704395e-05, 7.99284069e-05])
#yrms = np.array([9.65669677e-05, 5.61440470e-05, 4.89721770e-05, 4.80127438e-05, 5.86293035e-05])
#xrms_err = [0, 0, 0,0,0,0]
#yrms_err = [0, 0, 0,0,0,0]
#emittance = get_normemit(energy, quad_list, quad_list, np.array(xrms), np.array(yrms), np.array(xrms_err), np.array(yrms_err), adapt_ranges=False, num_points=len(quad_list), show_plots=True)
##########################################################################
#print(emittance/1e-6)

# Calculate new ranges and save them to a Matlab Array PV
new_x_range = adapt_range(quad_list,np.array(xrms),axis='x',num_points = len(quad_list))
print('X range adapted', new_x_range)
new_y_range = adapt_range(quad_list,np.array(yrms),axis='y',num_points = len(quad_list))
print('Y range adapted', new_y_range)

#print(emittance)
# Print min and max adapted quad ranges and num of points to matlab array PV
caput('SIOC:SYS1:ML01:AO651',np.min(new_x_range))
caput('SIOC:SYS1:ML01:AO652',np.max(new_x_range))
caput('SIOC:SYS1:ML01:AO653',np.min(new_y_range))
caput('SIOC:SYS1:ML01:AO654',np.max(new_y_range))
caput('SIOC:SYS1:ML01:AO655',len(new_x_range))




