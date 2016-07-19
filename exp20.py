from exp_helpers import base_experiment
from numpy import arange

# like 3 and 18, but filter down first 
base_experiment(expnum = 20, damages_values = arange(0,50,1), histogram_flag = 1, filter_type = "outside", max_trials = 1, sparsity_cutoff = 20.75)