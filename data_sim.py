from packages import *
from parameters import PARAMS
from nn import MODEL
from dataset import DATASET

class DATA_SIM():
    def __init__(self,g):
        self.g = g
    
    
    
data_sim = DATA_SIM(0)
data_sim.solution_plots()