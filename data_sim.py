from packages import *
from params import *
from nn import *

savePath = './.trained_model_params.pt'
model.eval()
model.load_state_dict(torch.load(savePath))

data = CustDataSet()
x,y = data.X,model(data.X)

