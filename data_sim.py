from packages import *
from params import *
from nn import *

savePrePath = './.pretrained_model_params.pt'
savePath = './.trained_model_params.pt'
model.eval()
model.load_state_dict(torch.load(savePath))

data = CustDataSet()
model.eval()
x,y_pred = data.X,model(data.X)

yLen = y_pred.shape[0]

#-----------------------------------------------------------------------
#Allocations/prices from predictions: TODAY
E = y_pred[...,equity] #equity
B = y_pred[...,bond] #bonds
P = y_pred[...,eqprice] #price of equity
Q = y_pred[...,bondprice] #price of bonds

#BC accounting: Consumption
E_ = padAssets(x[...,equity],yLen=yLen,side=0) #equity lag
B_ = padAssets(x[...,bond],yLen=yLen,side=0) #bond lag
y_ = x[...,incomes] #state-contingent endowment
Δ_ = x[...,divs] #state-contingent dividend
Chat = y_ + (P+Δ_)*E_ + B_ \
    - P*padAssets(E,yLen=yLen,side=1) - Q*padAssets(B,yLen=yLen,side=1)\
    - debtPay.flatten()*torch.ones(yLen,L*J).to(device) \
    - τ.flatten()*torch.ones(yLen,L*J).to(device) \
    - ϕ(padAssets(B,yLen=yLen,side=1))

#Penalty if Consumption is negative 
ϵc = 1e-12
C = torch.maximum(Chat,ϵc*(Chat*0+1))
cpen = -torch.sum(torch.less(Chat,0)*Chat/ϵc,-1)[:,None]

#-----------------------------------------------------------------------
#1-PERIOD FORECAST: for Euler expectations
#state variable construction 

#lagged asset holdings tomorrow are endog. asset holdings today 
endog = torch.concat([E,B],-1)[None].repeat(S,1,1) 
#state contingent vars
exog = torch.outer(shocks,
    torch.concat([y.flatten(),F.reshape(1),δ.reshape(1)],0))\
    [:,None,:].repeat(1,yLen,1) 
#full state variable tomorrow 
Xf = torch.concat([endog,exog],-1).float().detach()
#predictions for forecast values 
Yf = model(Xf).detach() 

#Allocations/prices from forecast predictions: TOMORROW (f := forecast)
Ef = Yf[...,equity] #equity forecast
Bf = Yf[...,bond] #bond forecast
Pf = Yf[...,eqprice] #equity price forecast
Qf = Yf[...,bondprice] #bond price forecast

#BC accounting: consumption 
#equity forecast lags
Ef_ = padAssets(E,yLen=yLen,side=0)[None].repeat(S,1,1) 
#bond forecase lags
Bf_ = padAssets(B,yLen=yLen,side=0)[None].repeat(S,1,1) 
#endowment realization
yf_ = Xf[...,incomes] 
#dividend realization
Δf_ = Xf[...,divs] 
#cons forecast
Cf = yf_ + (Pf+Δf_)*Ef_ + Bf_ \
    - Pf*padAssetsF(Ef,yLen=yLen,side=1) \
    - Qf*padAssetsF(Bf,yLen=yLen,side=1) \
    - debtPay.flatten()*torch.ones(S,yLen,L*J).to(device) \
    - τ.flatten()*torch.ones(S,yLen,L*J).to(device) \
    - ϕ(padAssetsF(Bf,yLen=yLen,side=1))

#Euler Errors: equity then bond: THIS IS JUST E[MR]-1=0
#FIX THE C,CF TO REMOVE YOUNGEST AND OLDEST
Ceuler = C.reshape(yLen,L,J)[...,:-1,:].reshape(yLen,(L-1)*J)
Cfeuler = Cf.reshape(S,yLen,L,J)[...,:-1,:].reshape(S,yLen,(L-1)*J)
eqEuler = torch.mean(
    torch.abs(
    upinv(β*torch.tensordot(up(Cfeuler)*(Pf+Δf_)
    ,torch.tensor(probs),dims=([0],[0]))/P)/Ceuler
    -1),
    -1
)[:,None]

bondEuler = torch.mean(
    torch.abs(
    upinv(β*torch.tensordot(up(Cfeuler)
    ,torch.tensor(probs),dims=([0],[0]))/Q)/Ceuler
    -1),
    -1
)[:,None]

#Market Clearing Errors
equityMCC = torch.abs(1-torch.sum(E,-1))[:,None]
bondMCC =   torch.abs(torch.sum(B,-1))[:,None]

#so in each period, the error is the sum of above
#EULERS + MCCs + consumption penalty 
loss_vec = torch.concat([eqEuler,bondEuler,equityMCC,bondMCC,cpen],-1)

#set model back to training (dropout back on)
model.train()

p=2
torch.sum(loss_vec**p,-1)**(1/p)