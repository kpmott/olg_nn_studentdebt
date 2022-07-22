from params import *
from packages import *
import detSS

#-------------------------------------------------------------------------------
"""
#this module defines - the neural network (with my custom loss function)
#                    - training data generation      
"""

#for use later: detSS allocs and prices
ebar,bbar,pbar,qbar,cbar = detSS.detSS_allocs()

#loss function for comparing "output" to labels
loss = nn.L1Loss()#nn.MSELoss(reduction='mean')

#This is a custom activation function for the output layer
class custAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    #here I define the activation function: it takes in [e,b,p,q]
    def forward(self,x):
        nn_eq = nn.Softmax(dim=-1) #since ownerships must sum to unity
        nn_bond = nn.Tanh() #roughly expected in [-1,1]?
        nn_prices = nn.Softplus() #strictly positive

        act1 = nn_eq(x[...,equity])
        act2 = nn_bond(x[...,bond])
        #act2 = act2 - torch.mean(act2)
        act3 = nn_prices(x[...,prices])
        
        x = torch.concat([act1,act2,act3],dim=-1).to(device)
        
        return x

#THIS IS THE NEURAL NETWORK!
class MODEL(nn.Module):
    
    #the __init__ function declares things for use later: like the model itself
    def __init__(self):
        #inherit all traits from the PyTorch Lightning Module stock 
        super().__init__()
        
        #Network architecture parameters
        sizes = [input,2048,2048,2048,output]
        
        #Network architecture 
        dp=0.0 #dropout parameter
        self.model = nn.Sequential(
            nn.Linear(in_features=sizes[0],out_features=sizes[1]),
            nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[1],out_features=sizes[2]),
            nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[2],out_features=sizes[3]),
            nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[3],out_features=sizes[4]),
            custAct() #output uses custom activation function 
        )
        
        #put on GPU
        self.model = self.model.to(device)
        
        #which loss function to use throughout: declared above 
        self.loss = loss
        
    #given x, how to form predictions 
    def forward(self,x):
        return self.model(x).to(device)
    
    #This is the ECONOMIC loss function: 
    #sum of Euler residuals + Market Clearing Conditions (MCCs)
    def losscalc(self,x):
        
        #make sure we're on GPU
        #self.model.to(device)
        
        #set to evaluation mode (turn dropout off)
        self.model.eval()
        
        #Given x, calculate predictions 
        y_pred = self.model(x)
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
            Yf = self.model(Xf)

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
        eqEuler = torch.mean(
            torch.abs(
            torch.tensordot(β*up(Cf[...,isNotYoungest])/up(C[...,isNotOldest])\
            *(Pf+Δf_)/P
            ,torch.tensor(probs),dims=([0],[0])) -1),
            -1
        )[:,None]

        bondEuler = torch.mean(
            torch.abs(
            torch.tensordot(β*up(Cf[...,isNotYoungest])/up(C[...,isNotOldest])\
            *1/Q
            ,torch.tensor(probs),dims=([0],[0])) -1),
            -1
        )[:,None]

        #Market Clearing Errors
        equityMCC = torch.abs(1-torch.sum(E,-1))[:,None]
        bondMCC =   torch.abs(torch.sum(B,-1))[:,None]

        #so in each period, the error is the sum of above
        #EULERS + MCCs + consumption penalty 
        loss_vec = torch.concat([eqEuler,bondEuler,equityMCC,bondMCC,cpen],-1)

        #set model back to training (dropout back on)
        self.model.train()

        p=2
        return torch.sum(loss_vec**p,-1)**(1/p)

    
#make sure we're on GPU 
model = MODEL()#.to(device)

#This generates the training data
class CustDataSet(Dataset):
    def __init__(self,pretrain=False):
        model.eval() #turn off dropout
        
        #ignore gradients: faster
        with torch.no_grad():
            
            #state variables: X
            #prediction variables: Y
            shist, zhist = SHOCKS() #shock path

            #Firms
            Yhist = F*zhist
            δhist = δ*zhist
            yhist = y*zhist

            #Network training data and predictions
            X = torch.zeros(T,input).to(device)
            Y = torch.zeros(T,output).to(device)
            X[0] = torch.concat(
                [ebar.flatten(),bbar.flatten(),
                yhist[0].flatten(),Yhist[0,0],δhist[0,0]],
                0)
            Y[0] = model(X[0])
            for t in range(1,T):
                X[t] = torch.concat(
                    [Y[t-1,equity],Y[t-1,bond],
                    yhist[t].flatten(),Yhist[t,0],δhist[t,0]],
                    0)
                Y[t] = model(X[t])
        
        #training inputs
        self.X = X[time]

        #training labels: if pretrain then detSS otherwise zeros
        if pretrain:
            self.Y = torch.ones(train,output).to(device)*\
                torch.concat(
                    [ebar.flatten(),bbar.flatten(),
                    pbar.flatten(),qbar.flatten()],
                    0
                ).detach().clone()
        else:
            self.Y = torch.zeros(train).float().to(device).detach().clone()

        model.train() #turn on dropout for training

    #mandatory to determine length of data
    def __len__(self):
        return len(self.Y)
    
    #return index batches
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]