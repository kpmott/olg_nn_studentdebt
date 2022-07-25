#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3
#-------------------------------------------------------------------------------

from packages import *
from params import *
from nn import *

savePrePath = './.pretrained_model_params.pt'
savePath = './.trained_model_params.pt'
model.eval()
model.load_state_dict(torch.load(savePath))

data = CustDataSet()
x = data.X
model.losscalc(x,full_loss=True)

#Data
model.eval()        
#Given x, calculate predictions 
y_pred = model(x)
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
ϵc = 1e-8
C = torch.maximum(Chat,ϵc*(Chat*0+1))
cpen = -torch.sum(torch.less(Chat,0)*Chat/ϵc,-1)[:,None]

#-----------------------------------------------------------------------
#1-PERIOD FORECAST: for Euler expectations
#state variable construction 

with torch.no_grad():
    #lagged asset holdings tomorrow are endog. asset holdings today 
    endog = torch.concat([E,B],-1)[None].repeat(S,1,1) 
    #state contingent vars
    exog = torch.outer(shocks,
        torch.concat([y.flatten(),F.reshape(1),δ.reshape(1)],0))\
        [:,None,:].repeat(1,yLen,1) 
    #full state variable tomorrow 
    Xf = torch.concat([endog,exog],-1).float().detach()
    #predictions for forecast values 
    Yf = model(Xf)

    #Allocations/prices from forecast predictions: 
    #TOMORROW (f := forecast)
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
eqEuler = torch.abs(
    β*torch.tensordot(up(Cf[...,isNotYoungest])*(Pf+Δf_)
    ,torch.tensor(probs),dims=([0],[0]))/(up(C[...,isNotOldest])*P) - 1.
)

bondEuler = torch.abs(
    β*torch.tensordot(up(Cf[...,isNotYoungest])
    ,torch.tensor(probs),dims=([0],[0]))/(up(C[...,isNotOldest])*Q) - 1.
)

#Market Clearing Errors
equityMCC = torch.abs(1-torch.sum(E,-1))[:,None]
bondMCC =   torch.abs(torch.sum(B,-1))[:,None]

#so in each period, the error is the sum of above
#EULERS + MCCs + consumption penalty 
loss_vec = torch.concat([eqEuler,bondEuler,equityMCC,bondMCC,cpen],-1)

#Plot globals
linestyle = ['-','--',':']
linecolor = ['k','b','r']
plottime = slice(-150,train,1)

#Consumption plot
cplot = C.reshape(train,L,J)[plottime]
for j in range(J):
    plt.plot(cplot[:,:,j].detach().cpu().t(),linestyle[j]+linecolor[j])
plt.xticks([i for i in range(L)]);plt.xlabel("i")
plt.title("Life-Cycle Consumption")
plt.legend(handles=
    [matplotlib.lines.Line2D([],[],
    linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
    for j in range(J)]
)
plt.savefig('.c.png');plt.clf()

#Bond plot
bplot = B.reshape(train,L-1,J)[plottime]
for j in range(J):
    plt.plot(bplot[:,:,j].detach().cpu().t(),linestyle[j]+linecolor[j])
plt.xticks([i for i in range(L-1)]);plt.xlabel("i")
plt.title("Life-Cycle Bonds")
plt.legend(handles=
    [matplotlib.lines.Line2D([],[],
    linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
    for j in range(J)]
)
plt.savefig('.b.png');plt.clf()

#Equity plot
eplot = E.reshape(train,L-1,J)[plottime]
for j in range(J):
    plt.plot(eplot[:,:,j].detach().cpu().t(),linestyle[j]+linecolor[j])
plt.xticks([i for i in range(L-1)]);plt.xlabel("i")
plt.title("Life-Cycle Equity")
plt.legend(handles=
    [matplotlib.lines.Line2D([],[],
    linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
    for j in range(J)]
)
plt.savefig('.e.png');plt.clf()

#Prices plot
pplot = P[plottime]
plt.plot(pplot.detach().cpu(),'k-')
plt.title('Equity Price')
plt.xticks([])
plt.savefig('.p.png');plt.clf()

#Bond plot
qplot = Q[plottime]
plt.plot(qplot.detach().cpu(),'k-')
plt.title('Bond Price')
plt.xticks([])
plt.savefig('.q.png');plt.clf()

#Excess return 
Δ = x[...,divs]
eqRet = annualize((P[1:] + Δ[1:])/P[:-1])
bondRet = annualize(1/Q[:-1])
exRet = eqRet-bondRet
exRetplot = exRet[plottime]
plt.plot(exRetplot.detach().cpu(),'k-')
plt.title('Excess Return')
plt.xticks([])
plt.savefig('.exret.png');plt.clf()
