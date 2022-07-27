from packages import *
from parameters import PARAMS

class CUSTACT(nn.Module):
    
    def __init__(self,g):
        super().__init__()
        self.params = PARAMS(g)
    
    #here I define the activation function: it takes in [e,b,p,q]
    def forward(self,x):
        self.params = self.params
        nn_eq = nn.Softmax(dim=-1) #since ownerships must sum to unity
        nn_bond = nn.Tanh() #roughly expected in [-1,1]?
        nn_prices = nn.Softplus() #strictly positive

        act1 = nn_eq(x[...,self.params.equity])
        act2 = nn_bond(x[...,self.params.bond])
        act3 = nn_prices(x[...,self.params.prices])
        
        x = torch.concat([act1,act2,act3],dim=-1).to(self.params.device)
        
        return x

class MODEL(nn.Module):
    
    #the __init__ function declares things for use later: like the model itself
    def __init__(self,g):
        self.params = PARAMS(g)
        
        #inherit all traits from the torch.mm module
        super().__init__()
        
        #Network architecture parameters
        sizes = [self.params.input,2048,2048,2048,self.params.output]
        
        #Network architecture 
        dp=0.175 #dropout parameter
        self.model = nn.Sequential(
            nn.Linear(in_features=sizes[0],out_features=sizes[1]),
            nn.ReLU(),nn.Dropout(p=0.),
            nn.Linear(in_features=sizes[1],out_features=sizes[2]),
            nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[2],out_features=sizes[3]),
            nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[3],out_features=sizes[4]),
            CUSTACT(g) #output uses custom activation function 
        )

        #put on GPU
        self.model = self.model.to(self.params.device)
        
        #which loss function to use throughout
        self.loss = nn.L1Loss()
        
    #given x, how to form predictions 
    def forward(self,x):
        return self.model(x).to(self.params.device)
    
    #This is the ECONOMIC loss function: 
    #sum of Euler residuals + Market Clearing Conditions (MCCs)
    def losscalc(self,x,full_loss=False):
        #set to evaluation mode (turn dropout off)
        self.model.eval()
        
        #Given x, calculate predictions 
        y_pred = self.model(x)
        yLen = y_pred.shape[0]

        #-----------------------------------------------------------------------
        #Allocations/prices from predictions: TODAY
        E = y_pred[...,self.params.equity] #equity
        B = y_pred[...,self.params.bond] #bonds
        P = y_pred[...,self.params.eqprice] #price of equity
        Q = y_pred[...,self.params.bondprice] #price of bonds

        #BC accounting: Consumption
        #equity lag
        E_ = self.params.padAssets(x[...,self.params.equity],yLen=yLen,side=0) 
        #bond lag
        B_ = self.params.padAssets(x[...,self.params.bond],yLen=yLen,side=0) 
        #state-contingent endowment
        y_ = x[...,self.params.incomes] 
        #state-contingent dividend
        Δ_ = x[...,self.params.divs] 
        #cons from BC
        Chat = y_ + (P+Δ_)*E_ + B_ \
            - P*self.params.padAssets(E,yLen=yLen,side=1) \
            - Q*self.params.padAssets(B,yLen=yLen,side=1) \
            - self.params.debtPay.flatten()*\
                torch.ones(yLen,self.params.L*self.params.J)\
                    .to(self.params.device) \
            - self.params.τ.flatten()\
                *torch.ones(yLen,self.params.L*self.params.J)\
                    .to(self.params.device) \
            - self.params.ϕ(self.params.padAssets(B,yLen=yLen,side=1))

        #Penalty if Consumption is negative 
        ϵc = 1e-8
        C = torch.maximum(Chat,ϵc*(Chat*0+1))
        cpen = -torch.sum(torch.less(Chat,0)*Chat/ϵc,-1)[:,None]

        #-----------------------------------------------------------------------
        #1-PERIOD FORECAST: for Euler expectations
        #state variable construction 
        
        with torch.no_grad():
            #lagged asset holdings tomorrow are endog. asset holdings today 
            endog = torch.concat([E,B],-1)[None].repeat(self.params.S,1,1) 
            #state contingent vars
            exog = torch.outer(self.params.shocks,
                torch.concat(
                    [self.params.y.flatten(),self.params.F.reshape(1),
                    self.params.δ.reshape(1)],0)
                )[:,None,:].repeat(1,yLen,1) 
            #full state variable tomorrow 
            Xf = torch.concat([endog,exog],-1).float().detach()
            #predictions for forecast values 
            Yf = self.model(Xf)

            #Allocations/prices from forecast predictions: 
            #TOMORROW (f := forecast)
            Ef = Yf[...,self.params.equity] #equity forecast
            Bf = Yf[...,self.params.bond] #bond forecast
            Pf = Yf[...,self.params.eqprice] #equity price forecast
            Qf = Yf[...,self.params.bondprice] #bond price forecast
            
            #BC accounting: consumption 
            #equity forecast lags
            Ef_ = self.params.padAssets(E,yLen=yLen,side=0)[None]\
                .repeat(self.params.S,1,1) 
            #bond forecase lags
            Bf_ = self.params.padAssets(B,yLen=yLen,side=0)[None]\
                .repeat(self.params.S,1,1) 
            #endowment realization
            yf_ = Xf[...,self.params.incomes] 
            #dividend realization
            Δf_ = Xf[...,self.params.divs] 
            #cons forecast
            Cf = yf_ + (Pf+Δf_)*Ef_ + Bf_ \
                - Pf*self.params.padAssetsF(Ef,yLen=yLen,side=1) \
                - Qf*self.params.padAssetsF(Bf,yLen=yLen,side=1) \
                - self.params.debtPay.flatten()*\
                    torch.ones(self.params.S,yLen,self.params.L*self.params.J)\
                        .to(self.params.device) \
                - self.params.τ.flatten()*\
                    torch.ones(self.params.S,yLen,self.params.L*self.params.J)\
                        .to(self.params.device) \
                - self.params.ϕ(self.params.padAssetsF(Bf,yLen=yLen,side=1))

        #Euler Errors: equity then bond: THIS IS JUST E[MR]-1=0
        eqEuler = torch.mean(torch.abs(
            self.params.β*torch.tensordot(
                self.params.up(Cf[...,self.params.isNotYoungest])*(Pf+Δf_),
                torch.tensor(self.params.probs),dims=([0],[0])
            )\
            /(self.params.up(C[...,self.params.isNotOldest])*P) - 1.
            ),-1
        )[:,None]

        bondEuler = torch.mean(torch.abs(
            self.params.β*torch.tensordot(
                self.params.up(Cf[...,self.params.isNotYoungest]),
                torch.tensor(self.params.probs),dims=([0],[0])
            )\
            /(self.params.up(C[...,self.params.isNotOldest])*Q) - 1.
            ),-1
        )[:,None]

        #Market Clearing Errors
        equityMCC = torch.abs(1-torch.sum(E,-1))[:,None]
        bondMCC   = torch.abs(torch.sum(B,-1))[:,None]

        #so in each period, the error is the sum of above
        #EULERS + MCCs + consumption penalty 
        loss_vec = torch.concat([eqEuler,bondEuler,equityMCC,bondMCC,cpen],-1)

        #set model back to training (dropout back on)
        #self.model.train()

        if full_loss:
            return loss_vec
        else:
            p=1.
            return torch.sum(loss_vec**p,-1)**(1/p)