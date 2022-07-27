#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3
#-------------------------------------------------------------------------------
from packages import argparse
from nn import MODEL
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
model = MODEL(g)
train = TRAIN(g,model,saveTrain=True)
train.train()
train.solution_plots()