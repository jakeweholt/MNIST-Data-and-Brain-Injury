from exp_helpers import base_experiment
from numpy import arange

base_experiment(expnum = 15, damages_values = arange(0,50,.25), histogram_flag = 1, filter_type = "inside", max_trials = 1)