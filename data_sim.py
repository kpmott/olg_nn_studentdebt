from packages import *
from parameters import PARAMS
from nn import MODEL
from detSS import DET_SS_ALLOCATIONS
from dataset import DATASET

#-------------------------------------------------------------------------------
model = MODEL(0)
params=PARAMS(0)

model.eval()
model.load_state_dict(torch.load(params.savePath+'/.trained_model_params.pt'))

#-------------------------------------------------------------------------------
#Data
data = DATASET(0)
x = data.X
model.eval()        

#Given x, calculate predictions 
with torch.no_grad():
    y_pred = model(x)
    yLen = y_pred.shape[0]

    #---------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------
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

    p=2
    lossval = torch.sum(loss_vec**p,-1)**(1/p)

#---------------------------------------------------------------------------
#Plots
#Plot globals
linestyle = ['-','--',':']
linecolor = ['k','b','r']
plottime = slice(-150,train,1)
figsize = (10,4)

#---------------------------------------------------------------------------
#Consumption lifecycle plot
Clife = torch.zeros(train-L,L,J)
Cj = C.reshape(train,J,L).permute(0,2,1)
for j in range(J):
    for t in range(train-L):
        for i in range(L):
            Clife[t,i,j] = Cj[t+i,i,j]

plt.figure(figsize=figsize)
for j in range(J):
    plt.plot(
        Clife[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
    )
plt.xticks([i for i in range(L)]);plt.xlabel("i")
plt.title("Life-Cycle Consumption")
plt.legend(handles=
    [matplotlib.lines.Line2D([],[],
    linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
    for j in range(J)]
)
plt.savefig(plotPath+'.c.png');plt.clf()
plt.close()

#Consumption plot
Cj = C.reshape(train,J,L)
plt.figure(figsize=figsize)
for j in range(J):
    plt.plot(
        Cj[plottime,j,:].detach().cpu(),linestyle[j]+linecolor[j]
    )
plt.xticks([]);plt.xlabel("t")
plt.title("Consumption")
plt.legend(handles=
    [matplotlib.lines.Line2D([],[],
    linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
    for j in range(J)]
)
plt.savefig(plotPath+'.consProcess.png');plt.clf()
plt.close()

#Bond plot
Blife = torch.zeros(train-L,L-1,J)
Bj = B.reshape(train,J,L-1).permute(0,2,1)
for j in range(J):
    for t in range(train-L):
        for i in range(L-1):
            Blife[t,i,j] = Bj[t+i,i,j]
plt.figure(figsize=figsize)
for j in range(J):
    plt.plot(
        Blife[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
    )
plt.xticks([i for i in range(L-1)]);plt.xlabel("i")
plt.title("Life-Cycle Bonds")
plt.legend(handles=
    [matplotlib.lines.Line2D([],[],
    linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
    for j in range(J)]
)
plt.savefig(plotPath+'.b.png');plt.clf()
plt.close()

#Equity plot
Elife = torch.zeros(train-L,L-1,J)
Ej = E.reshape(train,J,L-1).permute(0,2,1)
for j in range(J):
    for t in range(train-L):
        for i in range(L-1):
            Elife[t,i,j] = Ej[t+i,i,j]
plt.figure(figsize=figsize)
for j in range(J):
    plt.plot(
        Elife[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
    )
plt.xticks([i for i in range(L-1)]);plt.xlabel("i")
plt.title("Life-Cycle Equity")
plt.legend(handles=
    [matplotlib.lines.Line2D([],[],
    linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
    for j in range(J)]
)
plt.savefig(plotPath+'.e.png');plt.clf()
plt.close()

#Equity price plot
pplot = P[plottime]
plt.figure(figsize=figsize)
plt.plot(pplot.detach().cpu(),'k-')
plt.title('Equity Price')
plt.xticks([])
plt.savefig(plotPath+'.p.png');plt.clf()
plt.close()

#Bond price plot
qplot = Q[plottime]
plt.figure(figsize=figsize)
plt.plot(qplot.detach().cpu(),'k-')
plt.title('Bond Price')
plt.xticks([])
plt.savefig(plotPath+'.q.png');plt.clf()
plt.close()

#Excess return 
Δ = x[...,divs]
eqRet = annualize((P[1:] + Δ[1:])/P[:-1])
bondRet = annualize(1/Q[:-1])
exRet = eqRet-bondRet
exRetplot = exRet[plottime]
plt.figure(figsize=figsize)
plt.plot(exRetplot.detach().cpu(),'k-')
plt.title('Excess Return')
plt.xticks([])
plt.savefig(plotPath+'.exret.png');plt.clf()
plt.close()

#---------------------------------------------------------------------------
#Individual returns
rets = annualize(
    ((P[1:train-L,None]+Δ[1:train-L,None])*Elife[:-1] + Blife[:-1]) \
        / (P[:train-L-1,None]*Elife[:-1] + Q[:train-L-1,None]*Blife[:-1]) 
)
plt.figure(figsize=figsize)
for j in range(J):  
    plt.plot(
        rets[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
    )
plt.xticks([i for i in range(L-1)]);plt.xlabel("i")
plt.title("Life-Cycle Expected Portfolio Returns")
plt.legend(handles=
    [matplotlib.lines.Line2D([],[],
    linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
    for j in range(J)]
)
plt.savefig(plotPath+'.rets.png');plt.clf()
plt.close()

#---------------------------------------------------------------------------
#Portfolio shares
eqshare = P[:train-L-1,None]*Elife[:-1] \
    / (P[:train-L-1,None]*Elife[:-1] + Q[:train-L-1,None]*Blife[:-1]) 
plt.figure(figsize=figsize)
for j in range(J):  
    plt.plot(
        eqshare[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
    )
plt.xticks([i for i in range(L-1)]);plt.xlabel("i")
plt.title("Life-Cycle Portfolio Share: Equity Asset")
plt.legend(handles=
    [matplotlib.lines.Line2D([],[],
    linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
    for j in range(J)]
)
plt.savefig(plotPath+'.port.png');plt.clf()
plt.close()

#---------------------------------------------------------------------------
#Expected utility
EU = torch.mean(
    torch.tensordot(
        u(Clife),torch.tensor([β**i for i in range(L)]),dims=([1],[0])
    ),0
)
plt.figure(figsize=figsize)
plt.bar([j for j in range(J)],EU.cpu())
plt.xticks([j for j in range(J)]);plt.xlabel('j')
plt.title('Expected Utility')
plt.savefig(plotPath+'.EU.png');plt.clf();plt.close()