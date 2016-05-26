from exp_helpers import base_experiment
from numpy import arange

# as sanity check: this should be same as 4
base_experiment(expnum = 17, damages_values = arange(0,50,1), histogram_flag = 1, filter_type = "inside", max_trials = 1)