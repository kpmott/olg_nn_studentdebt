#!/home/kpmott/Git/olg.pytorch/pyt/bin/python3
#STILL TO DO
#------------------------------------------------------------------------------------------------------

from packages import *
from params import *
from nn import *

#------------------------------------------------------------------------------------------------------
#STEP 1: PRETRAINING

#generate pretraining data: labels are detSS
data = CustDataSet(pretrain=True) 

#load training data into DataLoader object for batching (ON GPU)
train_loader = DataLoader(data,batch_size=50,generator=torch.Generator(device="cuda"),shuffle=True)

#define and fit the trainer
trainer = pl.Trainer(max_epochs=5000, accelerator="gpu",logger=False,enable_checkpointing=False)
trainer.fit(model=model,train_dataloaders=train_loader)

#------------------------------------------------------------------------------------------------------
#STEP 2: MAIN TRAINING

#maximum iterations
iters = 15000

#list of losses to track over training cycle
losshist = np.zeros((iters))

#machine tolerance: end when loss is below this
ϵ = 1e-3

#training loop: Draw data from unique path of shocks. Train with batch_size=__ for one epoch. Then repeat
for thyme in tqdm(range(iters)):
    
    #draw data and load (ON GPU) into DataLoader for batching
    data = CustDataSet()
    train_loader = DataLoader(data,batch_size=32,generator=torch.Generator(device="cuda"),shuffle=True)
    
    #define trainer
    trainer = pl.Trainer(max_epochs=1, accelerator="gpu",logger=False,enable_checkpointing=False,enable_model_summary=False)
    trainer.fit(model=model,train_dataloaders=train_loader)
    
    #check loss, plot
    losses = model.losscalc(data.Σ)
    lossrun = loss(losses,losses*0).cpu().detach().numpy()
    losshist[thyme] = lossrun
    plt.plot(losshist[:thyme+1]);plt.yscale('log');plt.savefig('plot_losses.png');plt.clf()
    if lossrun < ϵ:
        print("Convergence in "+str(thyme)+" steps.")
        break

#------------------------------------------------------------------------------------------------------
#STEP 3: Analayze equilibrium 

#function for calcuations and plotting
def plots():
    #Inspect output
    #set to evaluation mode (turn dropout off)
    model.eval()
    Σ = data.Σ
    Y = model(Σ)
    
    #-------------------------------------------------------------------------------------
    #Allocations/prices from predictions: TODAY
    E = Y[...,equity] #equity
    B = Y[...,bond] #bonds
    P = Y[...,price] #price of equity
    Q = Y[...,ir] #price of bonds

    #BC accounting: Consumption
    E_ = torch.nn.functional.pad(Σ[...,equity],(1,0)) #equity lag: in state variable
    B_ = torch.nn.functional.pad(Σ[...,bond],(1,0)) #bond lag: in state variable
    Ω_ = Σ[...,endow] #state-contingent endowment
    Δ_ = Σ[...,div] #state-contingent dividend
    Chat = Ω_ + (P+Δ_)*E_ + B_ - P*torch.nn.functional.pad(E,(0,1)) - Q*torch.nn.functional.pad(B,(0,1)) #Budget Constraint
    
    #Penalty if Consumption is negative 
    ϵc = 1e-6
    C = torch.maximum(Chat,ϵc*(Chat*0+1))
    cpen = -torch.sum(torch.less(Chat,0)*Chat/ϵc)

    #-------------------------------------------------------------------------------------
    #1-PERIOD FORECAST: for Euler expectations
    
    #state variable construction 
    endog = torch.concat([E,B],-1)[None].repeat(S,1,1) #lagged asset holdings tomorrow are endog. asset holdings today 
    exog = torch.tensor([[*[wvec[s]],*ωvec[s], *[δvec[s]]] for s in range(S)])[:,None,:].repeat(1,len(Y),1) #state-contingent realizations tomorrow
    Σf = torch.concat([endog,exog],-1).float() #full state variable tomorrow 
    Yf = model(Σf) #predictions for forecast values
    
    #Allocations/prices from forecast predictions: TOMORROW (f stands for forecast)
    Ef = Yf[...,equity] #equity forecast
    Bf = Yf[...,bond] #bond forecast
    Pf = Yf[...,price] #equity price forecast
    Qf = Yf[...,ir] #bond price forecast
    
    #BC accounting: consumption 
    Ef_ = torch.nn.functional.pad(E,(1,0))[None].repeat(S,1,1) #equity lags: from today's calculation
    Bf_ = torch.nn.functional.pad(B,(1,0))[None].repeat(S,1,1) #bond lags
    Ωf_ = Σf[...,endow] #endowment realization
    Δf_ = Σf[...,div] #dividend realization
    Cf = Ωf_ + (Pf+Δf_)*Ef_ + Bf_ - Pf*torch.nn.functional.pad(Ef,(0,1,0,0,0,0)) - Qf*torch.nn.functional.pad(Bf,(0,1,0,0,0,0)) #budget constraint

    #Euler Errors: equity then bond: THIS IS JUST E[MR]-1=0
    eqEuler = torch.sum(torch.abs(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])*(Pf+Δf_)/P,torch.tensor(probs),dims=([0],[0]))-1),-1)
    bondEuler = torch.sum(torch.abs(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])/Q,torch.tensor(probs),dims=([0],[0]))-1),-1)

    #Market Clearing Errors
    equityMCC = torch.abs(equitysupply-torch.sum(E,-1))
    bondMCC = torch.abs(bondsupply-torch.sum(B,-1))

    #so in each period, the error is the sum of above
    #EULERS + MCCs + consumption penalty 
    loss_vec = eqEuler + bondEuler + equityMCC + bondMCC + cpen

    #set model back to training (dropout back on)
    model.train()


    #ANNUALIZED returns
    Δ = Σ[:,div] #dividends
    eqRet = ((P[1:] + Δ[1:])/P[:-1])**(L/60) - 1 #equity return 
    bondRet = (1/Q[1:])**(L/60) - 1 #bond return
    exRet = eqRet - bondRet #equity excess return

    #plots
    plt.plot(C[-100:].detach().cpu());plt.savefig('plot_cons.png');plt.clf()
    plt.plot(exRet[-100:].detach().cpu());plt.savefig('plot_exRet.png');plt.clf()
    plt.plot(P[-100:].detach().cpu());plt.savefig('plot_p.png');plt.clf()
    plt.plot(Q[-100:].detach().cpu());plt.savefig('plot_q.png');plt.clf()
    plt.plot(E[-100:].t().detach().cpu());plt.savefig('plot_e.png');plt.clf()
    plt.plot(B[-100:].t().detach().cpu());plt.savefig('plot_b.png');plt.clf()
    plt.plot(C[-100:].t().detach().cpu());plt.savefig('plot_c.png');plt.clf()

plots()