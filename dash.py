#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3
#-------------------------------------------------------------------------------

from packages import *
from params import *
from nn import *
from training import *

savePretrain = False
savePrePath = './.pretrained_model_params.pt'

saveTrain = True
savePath = './.trained_model_params.pt'

#-------------------------------------------------------------------------------
#STEP 1: PRETRAINING

if savePretrain:
    losses = pretrain_loop(epochs=1000,batchsize=50,lr=1e-4)
    #losses = pretrain_loop(epochs=1000,batchsize=50,lr=1e-5,losses=losses)
    #osses = pretrain_loop(epochs=1000,batchsize=100,lr=1e-6,losses=losses)
    #losses = pretrain_loop(epochs=1000,batchsize=175,lr=1e-6,losses=losses)

    model.eval()
    torch.save(model.state_dict(), savePrePath)
    model.train()

#-------------------------------------------------------------------------------
#STEP 2: MAIN TRAINING

if not savePretrain:
    model.load_state_dict(torch.load(savePrePath))

losses = train_loop(epochs=250,batchsize=32,lr=5e-7)
losses = train_loop(epochs=250,batchsize=32,lr=1e-7,losses=losses)
losses = train_loop(epochs=250,batchsize=100,lr=1e-7,losses=losses)

if saveTrain:
    model.eval()
    torch.save(model.state_dict(), savePath)
    model.train()