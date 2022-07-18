#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3
#-------------------------------------------------------------------------------

from packages import *
from params import *
from nn import *
from training import *

#-------------------------------------------------------------------------------
#STEP 1: PRETRAINING

pretrain_loop(250,1e-4)
pretrain_loop(250,1e-5)
pretrain_loop(250,1e-6)
pretrain_loop(250,1e-7)

#-------------------------------------------------------------------------------
#STEP 2: MAIN TRAINING

train_loop()