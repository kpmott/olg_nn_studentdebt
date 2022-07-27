#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3
#-------------------------------------------------------------------------------
from packages import *
from parameters import PARAMS
from nn import MODEL
from detSS import DET_SS_ALLOCATIONS
from dataset import DATASET
from training import TRAIN

# parser = argparse.ArgumentParser(
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter
# )
# parser.add_argument(
#     "-G", "--gpu", type=int, help="which gpu in [0,...,8]", default=0
# )
# parser.parse_args()
# args = parser.parse_args()
# config = vars(args)
# g = config['gpu']

saveTrain = True

#-------------------------------------------------------------------------------
# TRAINING
g=0
params = PARAMS(g)
train = TRAIN(g).train_loop
model = MODEL(g)

losses = train(epochs=1000,batchsize=32,lr=1e-5)
if saveTrain:
    model.eval()
    torch.save(model.state_dict(), params.savePath+'/.trained_model_params.pt')
    model.train()

losses = train(epochs=1000,batchsize=64,lr=1e-6,losses=losses)
if saveTrain:
    model.eval()
    torch.save(model.state_dict(), params.savePath+'/.trained_model_params.pt')
    model.train()

losses = train(epochs=1000,batchsize=100,lr=1e-7,losses=losses)
if saveTrain:
    model.eval()
    torch.save(model.state_dict(), params.savePath+'/.trained_model_params.pt')
    model.train()