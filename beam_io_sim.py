#NN Surrogate model class
from injector_surrogate_quads import *

from sampling_functions import get_ground_truth, get_beamsize
from emittance_calc import *
sys.path.append('../configs')
#Sim reference point to optimize around
from ref_config import ref_point

Model = Surrogate_NN()

Model.load_saved_model(model_path = '../models/', \
                       model_name = 'model_OTR2_NA_rms_emit_elu_2021-07-27T19_54_57-07_00')
Model.load_scaling()
Model.take_log_out = False

energy = 0.135

def get_sizes(quad):
    return get_beamsize(Model, ref_point, 0.5657 , -0.01063 ,-0.01  , quad)