from params import *
from packages import *
import detSS

#---------------------------------------------------------------------------------------------
"""
#this module defines - the neural network (with my custom loss function)
#                    - training data generation      
"""

#for use later: detSS allocs and prices
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

#loss function for comparing "output" to labels
loss = nn.MSELoss()

#This is a custom activation function for the output layer
class custAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    #here I define the activation function: it takes in [e,b,p,q]
    #e,b∈[0,1] (Tanh)
    #p,q>0 (SoftPlus)
    def forward(self,x):
        nn_tanh = nn.Tanh()
        nn_sp = nn.Softplus()
        
        act1 = nn_tanh(x[...,equity])
        act2 = nn_tanh(x[...,bond])
        act3 = nn_sp(x[...,price])
        act4 = nn_sp(x[...,ir])
        
        return torch.concat([act1,act2,act3,act4],dim=-1)

#THIS IS THE NEURAL NETWORK!
class MODEL(pl.LightningModule):
    
    #the __init__ function declares things for use later: like the model itself
    def __init__(self):
        #inherit all traits from the PyTorch Lightning Module stock 
        super().__init__()
        
        #Network architecture parameters
        sizes = [input,2048,2048,1024,1024,output]
        
        #Network architecture 
        dp=0.2 #dropout parameter
        self.model = nn.Sequential(
            nn.Linear(in_features=sizes[0],out_features=sizes[1]),nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[1],out_features=sizes[2]),nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[2],out_features=sizes[3]),nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[3],out_features=sizes[4]),nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[4],out_features=sizes[5]),custAct() #output uses custom activation function 
        )
        
        #which loss function to use throughout: declared above 
        self.loss = loss
        
    #given x, how to form predictions 
    def forward(self,x):
        return self.model(x)
    
    #This is the ECONOMIC loss function: sum of Euler residuals + Market Clearing Conditions (MCCs)
    def losscalc(self,x):
        
        #make sure we're on GPU
        self.model.to("cuda")
        
        #set to evaluation mode (turn dropout off)
        self.model.eval()
        
        #Given x, calculate predictions 
        y_pred = self.model(x)

        #-------------------------------------------------------------------------------------
        #Allocations/prices from predictions: TODAY
        E = y_pred[...,equity] #equity
        B = y_pred[...,bond] #bonds
        P = y_pred[...,price] #price of equity
        Q = y_pred[...,ir] #price of bonds

        #BC accounting: Consumption
        E_ = torch.nn.functional.pad(x[...,equity],(1,0)) #equity lag: in state variable
        B_ = torch.nn.functional.pad(x[...,bond],(1,0)) #bond lag: in state variable
        Ω_ = x[...,endow] #state-contingent endowment
        Δ_ = x[...,div] #state-contingent dividend
        Chat = Ω_ + (P+Δ_)*E_ + B_ - P*torch.nn.functional.pad(E,(0,1)) - Q*torch.nn.functional.pad(B,(0,1)) #Budget Constraint
        
        #Penalty if Consumption is negative 
        ϵc = 1e-6
        C = torch.maximum(Chat,ϵc*(Chat*0+1))
        cpen = -torch.sum(torch.less(Chat,0)*Chat/ϵc) 

        #-------------------------------------------------------------------------------------
        #1-PERIOD FORECAST: for Euler expectations
        
        #state variable construction 
        endog = torch.concat([E,B],-1)[None].repeat(S,1,1) #lagged asset holdings tomorrow are endog. asset holdings today 
        exog = torch.tensor([[*[wvec[s]],*ωvec[s], *[δvec[s]]] for s in range(S)])[:,None,:].repeat(1,len(y_pred),1) #state-contingent realizations tomorrow
        Σf = torch.concat([endog,exog],-1).float() #full state variable tomorrow 
        Yf = self.model(Σf).detach() #predictions for forecast values 

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
        eqEuler = torch.sum(torch.abs(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])*(Pf+Δf_)/P,torch.tensor(probs),dims=([0],[0]))-1),-1) +1
        #torch.prod(torch.abs(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])*(Pf+Δf_)/P,torch.tensor(probs),dims=([0],[0]))-1)+1,-1)
        bondEuler = torch.prod(torch.abs(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])/Q,torch.tensor(probs),dims=([0],[0]))-1),-1) +1
        #torch.prod(torch.abs(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])/Q,torch.tensor(probs),dims=([0],[0]))-1)+1,-1)

        #Market Clearing Errors
        equityMCC = torch.abs(equitysupply-torch.sum(E,-1))+1
        bondMCC = torch.abs(bondsupply-torch.sum(B,-1))+1

        #so in each period, the error is the sum of above
        #EULERS + MCCs + consumption penalty 
        loss_vec = eqEuler + bondEuler + equityMCC + bondMCC + cpen

        #set model back to training (dropout back on)
        self.model.train()

        return loss_vec

    #calculate the loss value
    def training_step(self,batch,batch_idx):
        
        #pass in x and y
        #x: input data
        #y: training "labels" (more on this below)
        x, y = batch
        
        #predictions from model 
        y_pred = self.model(x)
        
        #Two modes: when pretraining: want model to return detSS values ∀ input
        pretrain = (torch.max(y) > 0.)
        if pretrain: 
            #if we're pretraining: distance between model predictions (y_pred) and detSS (input with batch as y)
            loss = self.loss(y_pred,y)
        else:
            #"real" training: labels are zero because Equilibrium Residuals should be 0
            #in this case: I feed in y with batch as zeros 
            EQ_LOSS = self.losscalc(x) #equilibrium residuals
            loss = self.loss(EQ_LOSS,y)
        
        #return the right loss 
        return loss

    #define a forward step with predictions
    def predict_step(self,batch,batch_idx):
        X_batch, Y_batch = batch
        preds = self.model(X_batch.float())
        return preds
    
    #define the optimizer 
    def configure_optimizers(self):
        #x,y = batch
        #pretrain = (torch.max(y) > 0.)
        #if pretrain:
        #    lr=1e-4
        #else:
        #    lr=1e-7
        lr=1e-7
        return Adam(self.model.parameters(), lr=lr)

#make sure we're on GPU 
model = MODEL().to("cuda")

#This generates the training data
class CustDataSet(Dataset):
    def __init__(self,pretrain=False):
        model.to("cuda") #make sure we're on GPU
        model.eval() #turn off dropout
        
        #state variables: Σ
        #prediction variables: Y
        shist, whist, Ωhist, Δhist, Ω, Δ = SHOCKS() #shock path
        Σ = torch.zeros(T,input)
        Y = torch.zeros(T,output)
        Σ[0] = torch.tensor([*ebar,*bbar,*[wbar],*ω_scalar,*[δ_scalar]])
        Y[0] = model(Σ[0])
        for t in range(1,T):
            Σ[t] = torch.tensor([[*Y[t-1,equity],*Y[t-1,bond],*[whist[t]],*Ωhist[t],*[Δhist[t]]]])
            Y[t] = model(Σ[t])
        
        #lop off burn period
        Σ, Y = Σ[time], Y[time]
        
        #training inputs
        self.Σ = Σ.detach().clone()

        #training labels: if pretrain then detSS otherwise zeros
        if pretrain:
            self.Y = torch.ones(train,output)*torch.tensor([*ebar,*bbar,*[pbar],*[qbar]]).float()
        else:
            self.Y = torch.zeros(train).float()

        model.train() #turn on dropout for training

    #mandatory to determine length of data
    def __len__(self):
        return len(self.Y)
    
    #return index batches
    def __getitem__(self,idx):
        return self.Σ[idx], self.Y[idx]