from exp_helpers import base_experiment
from numpy import tile

base_experiment(expnum = 7, aging_flag = 1, damages_values = tile(.01, 100))