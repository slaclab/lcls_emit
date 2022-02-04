import sys
sys.path.append("../lcls_cu_injector_ml_model/configs/") 
sys.path.append("../lcls_cu_injector_ml_model/models/")
sys.path.append("../lcls_cu_injector_ml_model/injector_surrogate/") 


#Sim reference point to optimize around
from ref_config import ref_point

#NN Surrogate model class
from injector_surrogate_quads import *

from sampling_functions import get_ground_truth, get_beamsize
from emittance_calc import *

m_0 = 0.511*1e-3 # mass in [GeV]
d = 2.26 # [m] distance between Q525 and OTR2
l = 0.108 # effective length [m]

Model = Surrogate_NN()

Model.load_saved_model(model_path = '/Users/smiskov/Documents/SLAC/lcls_cu_injector_ml_model/models/', \
                       model_name = 'model_OTR2_NA_rms_emit_elu_2021-07-27T19_54_57-07_00')
Model.load_scaling()
Model.take_log_out = False

energy = 0.135

def get_beamsizes(quad):
    return get_beamsize(Model, ref_point, 0.5657 , -0.01063 ,-0.01  , quad)