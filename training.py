from packages import *
from parameters import PARAMS
#from nn import MODEL
from detSS import DET_SS_ALLOCATIONS
from dataset import DATASET
        
class TRAIN():
    def __init__(self,g,model,saveTrain=True):
        self.params = PARAMS(g)
        self.model = model#MODEL(g)
        self.dataset = DATASET(g,self.model)
        self.saveTrain = saveTrain
        self.g = g 

    def train_loop(self,epochs=100,batchsize=32,lr=1e-8,losses=[]):
        tol = 1e-3

        for epoch in tqdm(range(epochs)):
            #generate pretraining data: labels are detSS
            data = self.dataset

            #load training data into DataLoader object for batching (ON GPU)
            train_loader = DataLoader(
                data,batch_size=batchsize,
                generator=torch.Generator(device=self.params.device),
                shuffle=True,num_workers=0
            )

            #optimizer
            optimizer = Adam(self.model.parameters(), lr=lr)

            #actual loop
            batchloss = []
            for batch, (X,y) in enumerate(train_loader):
                    
                optimizer.zero_grad()
                lossval = self.model.loss(self.model.losscalc(X),y)
                batchloss.append(lossval.item())
                lossval.backward()#create_graph=True)
                optimizer.step()
            
            epochloss = np.mean(batchloss)
            losses.append(epochloss)
            
            figsize = (10,4)
            plt.figure(figsize=figsize)
            plt.plot(losses)
            plt.yscale('log')
            plt.title("Losses: "+"{:.2e}".format(losses[-1]))
            plt.xlabel("Epoch")
            plt.savefig(self.params.plotPath+'.plot_losses.png')
            plt.clf()
            plt.close()

            if epochloss < tol:
                break
            
            torch.cuda.empty_cache()
        
        return losses
    
    def train(self):
        
        num_regimes = 3
        epochs = [250,500,100]#[500 for epoch in range(num_regimes)]
        batches = [2**(1+5+batch) for batch in range(num_regimes)]
        lrs = [1e-5*10**(-n) for n in range(num_regimes)]

        losses=[]
        for regime in range(num_regimes):
            losses = self.train_loop(
                    epochs=epochs[regime],
                    batchsize=batches[regime],
                    lr=lrs[regime],
                    losses=losses
            )

            if self.saveTrain:
                self.model.eval()
                torch.save(self.model.state_dict(), 
                    self.params.savePath+'/.trained_model_params.pt'
                )
                self.model.train()

    def solution_plots(self):
        params = self.params
        model = self.model

        #model.eval()
        #model.load_state_dict(torch.load(params.savePath+'/.trained_model_params.pt'))

        #-------------------------------------------------------------------------------
        #Data
        model.eval()        
        data = DATASET(self.g,model)
        x = data.X

        #Given x, calculate predictions 
        with torch.no_grad():
            y_pred = model(x)
            yLen = y_pred.shape[0]

            #-----------------------------------------------------------------------
            #Allocations/prices from predictions: TODAY
            E = y_pred[...,params.equity] #equity
            B = y_pred[...,params.bond] #bonds
            P = y_pred[...,params.eqprice] #price of equity
            Q = y_pred[...,params.bondprice] #price of bonds

            #BC accounting: Consumption
            #equity lag
            E_ = params.padAssets(x[...,params.equity],yLen=yLen,side=0) 
            #bond lag
            B_ = params.padAssets(x[...,params.bond],yLen=yLen,side=0) 
            #state-contingent endowment
            y_ = x[...,params.incomes] 
            #state-contingent dividend
            Δ_ = x[...,params.divs] 
            #cons from BC
            Chat = y_ + (P+Δ_)*E_ + B_ \
                - P*params.padAssets(E,yLen=yLen,side=1) \
                - Q*params.padAssets(B,yLen=yLen,side=1) \
                - params.debtPay.flatten()*\
                    torch.ones(yLen,params.L*params.J)\
                        .to(params.device) \
                - params.τ.flatten()\
                    *torch.ones(yLen,params.L*params.J)\
                        .to(params.device) \
                - params.ϕ(params.padAssets(B,yLen=yLen,side=1))

            #Penalty if Consumption is negative 
            ϵc = 1e-8
            C = torch.maximum(Chat,ϵc*(Chat*0+1))
            cpen = -torch.sum(torch.less(Chat,0)*Chat/ϵc,-1)[:,None]

            #-----------------------------------------------------------------------
            #1-PERIOD FORECAST: for Euler expectations
            #state variable construction 
            
            
            #lagged asset holdings tomorrow are endog. asset holdings today 
            endog = torch.concat([E,B],-1)[None].repeat(params.S,1,1) 
            #state contingent vars
            exog = torch.outer(params.shocks,
                torch.concat(
                    [params.y.flatten(),params.F.reshape(1),
                    params.δ.reshape(1)],0)
                )[:,None,:].repeat(1,yLen,1) 
            #full state variable tomorrow 
            Xf = torch.concat([endog,exog],-1).float().detach()
            #predictions for forecast values 
            Yf = model(Xf)

            #Allocations/prices from forecast predictions: 
            #TOMORROW (f := forecast)
            Ef = Yf[...,params.equity] #equity forecast
            Bf = Yf[...,params.bond] #bond forecast
            Pf = Yf[...,params.eqprice] #equity price forecast
            Qf = Yf[...,params.bondprice] #bond price forecast
            
            #BC accounting: consumption 
            #equity forecast lags
            Ef_ = params.padAssets(E,yLen=yLen,side=0)[None]\
                .repeat(params.S,1,1) 
            #bond forecase lags
            Bf_ = params.padAssets(B,yLen=yLen,side=0)[None]\
                .repeat(params.S,1,1) 
            #endowment realization
            yf_ = Xf[...,params.incomes] 
            #dividend realization
            Δf_ = Xf[...,params.divs] 
            #cons forecast
            Cf = yf_ + (Pf+Δf_)*Ef_ + Bf_ \
                - Pf*params.padAssetsF(Ef,yLen=yLen,side=1) \
                - Qf*params.padAssetsF(Bf,yLen=yLen,side=1) \
                - params.debtPay.flatten()*\
                    torch.ones(params.S,yLen,params.L*params.J)\
                        .to(params.device) \
                - params.τ.flatten()*\
                    torch.ones(params.S,yLen,params.L*params.J)\
                        .to(params.device) \
                - params.ϕ(params.padAssetsF(Bf,yLen=yLen,side=1))

            #Euler Errors: equity then bond: THIS IS JUST E[MR]-1=0
            eqEuler = torch.mean(torch.abs(
                params.β*torch.tensordot(
                    params.up(Cf[...,params.isNotYoungest])*(Pf+Δf_),
                    torch.tensor(params.probs),dims=([0],[0])
                )\
                /(params.up(C[...,params.isNotOldest])*P) - 1.
                ),-1
            )[:,None]

            bondEuler = torch.mean(torch.abs(
                params.β*torch.tensordot(
                    params.up(Cf[...,params.isNotYoungest]),
                    torch.tensor(params.probs),dims=([0],[0])
                )\
                /(params.up(C[...,params.isNotOldest])*Q) - 1.
                ),-1
            )[:,None]

            #Market Clearing Errors
            equityMCC = torch.abs(1-torch.sum(E,-1))[:,None]
            bondMCC   = torch.abs(torch.sum(B,-1))[:,None]

            #so in each period, the error is the sum of above
            #EULERS + MCCs + consumption penalty 
            loss_vec = torch.concat([eqEuler,bondEuler,equityMCC,bondMCC,cpen],-1)

        #---------------------------------------------------------------------------
        #Plots
        #Plot globals
        linestyle = ['-','--',':']
        linecolor = ['k','b','r']
        plottime = slice(-150,params.train,1)
        figsize = (10,4)

        #---------------------------------------------------------------------------
        #Consumption lifecycle plot
        Clife = torch.zeros(params.train-params.L,params.L,params.J)
        Cj = C.reshape(params.train,params.J,params.L).permute(0,2,1)
        for j in range(params.J):
            for t in range(params.train-params.L):
                for i in range(params.L):
                    Clife[t,i,j] = Cj[t+i,i,j]

        plt.figure(figsize=figsize)
        for j in range(params.J):
            plt.plot(
                Clife[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
            )
        plt.xticks([i for i in range(params.L)]);plt.xlabel("i")
        plt.title("Life-Cycle Consumption")
        plt.legend(handles=
            [matplotlib.lines.Line2D([],[],
            linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
            for j in range(params.J)]
        )
        plt.savefig(params.plotPath+'.c.png');plt.clf()
        plt.close()

        #Consumption plot
        Cj = C.reshape(params.train,params.J,params.L)
        plt.figure(figsize=figsize)
        for j in range(params.J):
            plt.plot(
                Cj[plottime,j,:].detach().cpu(),linestyle[j]+linecolor[j]
            )
        plt.xticks([]);plt.xlabel("t")
        plt.title("Consumption")
        plt.legend(handles=
            [matplotlib.lines.Line2D([],[],
            linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
            for j in range(params.J)]
        )
        plt.savefig(params.plotPath+'.consProcess.png');plt.clf()
        plt.close()

        #Bond plot
        Blife = torch.zeros(params.train-params.L,params.L-1,params.J)
        Bj = B.reshape(params.train,params.J,params.L-1).permute(0,2,1)
        for j in range(params.J):
            for t in range(params.train-params.L):
                for i in range(params.L-1):
                    Blife[t,i,j] = Bj[t+i,i,j]
        plt.figure(figsize=figsize)
        for j in range(params.J):
            plt.plot(
                Blife[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
            )
        plt.xticks([i for i in range(params.L-1)]);plt.xlabel("i")
        plt.title("Life-Cycle Bonds")
        plt.legend(handles=
            [matplotlib.lines.Line2D([],[],
            linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
            for j in range(params.J)]
        )
        plt.savefig(params.plotPath+'.b.png');plt.clf()
        plt.close()

        #Equity plot
        Elife = torch.zeros(params.train-params.L,params.L-1,params.J)
        Ej = E.reshape(params.train,params.J,params.L-1).permute(0,2,1)
        for j in range(params.J):
            for t in range(params.train-params.L):
                for i in range(params.L-1):
                    Elife[t,i,j] = Ej[t+i,i,j]
        plt.figure(figsize=figsize)
        for j in range(params.J):
            plt.plot(
                Elife[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
            )
        plt.xticks([i for i in range(params.L-1)]);plt.xlabel("i")
        plt.title("Life-Cycle Equity")
        plt.legend(handles=
            [matplotlib.lines.Line2D([],[],
            linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
            for j in range(params.J)]
        )
        plt.savefig(params.plotPath+'.e.png');plt.clf()
        plt.close()

        #Equity price plot
        pplot = P[plottime]
        plt.figure(figsize=figsize)
        plt.plot(pplot.detach().cpu(),'k-')
        plt.title('Equity Price')
        plt.xticks([])
        plt.savefig(params.plotPath+'.p.png');plt.clf()
        plt.close()

        #Bond price plot
        qplot = Q[plottime]
        plt.figure(figsize=figsize)
        plt.plot(qplot.detach().cpu(),'k-')
        plt.title('Bond Price')
        plt.xticks([])
        plt.savefig(params.plotPath+'.q.png');plt.clf()
        plt.close()

        #Excess return 
        Δ = x[...,params.divs]
        eqRet = params.annualize((P[1:] + Δ[1:])/P[:-1])
        bondRet = params.annualize(1/Q[:-1])
        exRet = eqRet-bondRet
        exRetplot = exRet[plottime]
        plt.figure(figsize=figsize)
        plt.plot(exRetplot.detach().cpu(),'k-')
        plt.title('Excess Return')
        plt.xticks([])
        plt.savefig(params.plotPath+'.exret.png');plt.clf()
        plt.close()

        #---------------------------------------------------------------------------
        #Individual returns
        rets = params.annualize(
            ((P[1:params.train-params.L,None]+Δ[1:params.train-params.L,None])*Elife[:-1] + Blife[:-1]) \
                / (P[:params.train-params.L-1,None]*Elife[:-1] + Q[:params.train-params.L-1,None]*Blife[:-1]) 
        )
        plt.figure(figsize=figsize)
        for j in range(params.J):  
            plt.plot(
                rets[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
            )
        plt.xticks([i for i in range(params.L-1)]);plt.xlabel("i")
        plt.title("Life-Cycle Expected Portfolio Returns")
        plt.legend(handles=
            [matplotlib.lines.Line2D([],[],
            linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
            for j in range(params.J)]
        )
        plt.savefig(params.plotPath+'.rets.png');plt.clf()
        plt.close()

        #---------------------------------------------------------------------------
        #Portfolio shares
        eqshare = P[:params.train-params.L-1,None]*Elife[:-1] \
            / (P[:params.train-params.L-1,None]*Elife[:-1] + Q[:params.train-params.L-1,None]*Blife[:-1]) 
        plt.figure(figsize=figsize)
        for j in range(params.J):  
            plt.plot(
                eqshare[plottime,:,j].detach().cpu().t(),linestyle[j]+linecolor[j]
            )
        plt.xticks([i for i in range(params.L-1)]);plt.xlabel("i")
        plt.title("Life-Cycle Portfolio Share: Equity Asset")
        plt.legend(handles=
            [matplotlib.lines.Line2D([],[],
            linestyle=linestyle[j], color=linecolor[j], label=str(j)) 
            for j in range(params.J)]
        )
        plt.savefig(params.plotPath+'.port.png');plt.clf()
        plt.close()

        #---------------------------------------------------------------------------
        #Expected utility
        EU = torch.mean(
            torch.tensordot(
                params.u(Clife),torch.tensor([params.β**i for i in range(params.L)]),dims=([1],[0])
            ),0
        )
        plt.figure(figsize=figsize)
        plt.bar([j for j in range(params.J)],EU.cpu())
        plt.xticks([j for j in range(params.J)]);plt.xlabel('j')
        plt.title('Expected Utility')
        plt.savefig(params.plotPath+'.EU.png');plt.clf();plt.close()