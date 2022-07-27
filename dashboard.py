#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3
#-------------------------------------------------------------------------------
from packages import *
from parameters import PARAMS
from nn import MODEL
from detSS import DET_SS_ALLOCATIONS
from dataset import DATASET
from training import TRAIN

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-G", "--gpu", type=int, help="which gpu in [0,...,8]", default=0
)
parser.parse_args()
args = parser.parse_args()
config = vars(args)
g = config['gpu']

#-------------------------------------------------------------------------------
# TRAINING
train = TRAIN(g,saveTrain=True)
train.train()

from data_sim import DATA_SIM
data_sim = DATA_SIM(g)
data_sim.solution_plots()