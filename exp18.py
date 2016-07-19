from exp_helpers import base_experiment
from numpy import arange

# as sanity check: this should be same as 3
base_experiment(expnum = 18, damages_values = arange(0,50,1), histogram_flag = 1, filter_type = "outside", max_trials = 1)