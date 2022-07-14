from params import *
from packages import *
import detSS

#---------------------------------------------------------------------------------------------
"""
#this module defines - the neural network (with my custom loss function)
#                    - training data generation      
"""

#for use later: detSS allocs and prices
ebar,bbar,pbar,qbar,cbar = detSS.detSS_allocs()

#loss function for comparing "output" to labels
loss = nn.MSELoss(reduction='sum')

#This is a custom activation function for the output layer
class custAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    #here I define the activation function: it takes in [e,b,p,q]
    #e,b∈[0,1] (Tanh)
    #p,q>0 (SoftPlus)
    def forward(self,x):
        nn_sig = nn.Sigmoid()
        nn_tanh = nn.Tanh()
        nn_sp = nn.Softplus()

        act1 = nn_sig(x[...,equity])
        act2 = nn_tanh(x[...,bond])
        act3 = nn_sp(x[...,prices])
        
        x = torch.concat([act1,act2,act3],dim=-1)
        
        return x

#THIS IS THE NEURAL NETWORK!
class MODEL(pl.LightningModule):
    
    #the __init__ function declares things for use later: like the model itself
    def __init__(self):
        #inherit all traits from the PyTorch Lightning Module stock 
        super().__init__()
        
        #Network architecture parameters
        sizes = [input,2048,1024,output]
        
        #Network architecture 
        dp=0.05 #dropout parameter
        self.model = nn.Sequential(
            nn.Linear(in_features=sizes[0],out_features=sizes[1]),nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[1],out_features=sizes[2]),nn.ReLU(),nn.Dropout(p=dp),
            #nn.Linear(in_features=sizes[2],out_features=sizes[3]),nn.ReLU(),nn.Dropout(p=dp),
            #nn.Linear(in_features=sizes[3],out_features=sizes[4]),nn.ReLU(),nn.Dropout(p=dp),
            nn.Linear(in_features=sizes[2],out_features=sizes[3]),custAct() #output uses custom activation function 
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
        yLen = y_pred.shape[0]

        #-------------------------------------------------------------------------------------
        def padAssets(ASSETS,side=0):
            ASSETSJ = ASSETS.reshape(yLen,L-1,J)
            if side==0:
                ASSETSJpad = nn.functional.pad(ASSETSJ,(1,0))
            else:
                ASSETSJpad = nn.functional.pad(ASSETSJ,(0,1))
            return torch.flatten(ASSETSJpad,start_dim=-2)

        #Allocations/prices from predictions: TODAY
        E = y_pred[...,equity] #equity
        B = y_pred[...,bond] #bonds
        P = y_pred[...,eqprice] #price of equity
        Q = y_pred[...,bondprice] #price of bonds

        #BC accounting: Consumption
        E_ = padAssets(x[...,equity],side=0) #equity lag: in state variable
        B_ = padAssets(x[...,bond],side=0) #bond lag: in state variable
        y_ = x[...,incomes] #state-contingent endowment
        Δ_ = x[...,divs] #state-contingent dividend
        Chat = y_ + (P+Δ_)*E_ + B_ - P*padAssets(E,side=1) - Q*padAssets(B,side=1) \
            - debtPay.flatten()*torch.ones(yLen,L*J) - τ.flatten()*torch.ones(yLen,L*J) \
            - ϕ(padAssets(B,side=1))

        #Penalty if Consumption is negative 
        ϵc = 1e-6
        C = torch.maximum(Chat,ϵc*(Chat*0+1))
        cpen = -torch.sum(torch.less(Chat,0)*Chat/ϵc) 

        #-------------------------------------------------------------------------------------
        #1-PERIOD FORECAST: for Euler expectations
        def padAssetsF(ASSETS,side=0):
            ASSETSJ = ASSETS.reshape(S,yLen,L-1,J)
            if side==0:
                ASSETSJpad = nn.functional.pad(ASSETSJ,(1,0))
            else:
                ASSETSJpad = nn.functional.pad(ASSETSJ,(0,1))
            return torch.flatten(ASSETSJpad,start_dim=-2)

        #state variable construction 
        endog = torch.concat([E,B],-1)[None].repeat(S,1,1) #lagged asset holdings tomorrow are endog. asset holdings today 
        exog = torch.outer(shocks,torch.concat([y.flatten(),F.reshape(1),δ.reshape(1)],0))[:,None,:].repeat(1,yLen,1) #state contingent vars
        Xf = torch.concat([endog,exog],-1).float() #full state variable tomorrow 
        Yf = self.model(Xf).detach() #predictions for forecast values 

        #Allocations/prices from forecast predictions: TOMORROW (f stands for forecast)
        Ef = Yf[...,equity] #equity forecast
        Bf = Yf[...,bond] #bond forecast
        Pf = Yf[...,eqprice] #equity price forecast
        Qf = Yf[...,bondprice] #bond price forecast
        
        #BC accounting: consumption 
        Ef_ = padAssets(E,side=0)[None].repeat(S,1,1) #equity forecast lags
        Bf_ = padAssets(B,side=0)[None].repeat(S,1,1) #bond forecase lags
        yf_ = Xf[...,incomes] #endowment realization
        Δf_ = Xf[...,divs] #dividend realization
        Cf = yf_ + (Pf+Δf_)*Ef_ + Bf_ - Pf*padAssetsF(Ef,side=1) - Qf*padAssetsF(Bf,side=1) \
            - debtPay.flatten()*torch.ones(S,yLen,L*J) - τ.flatten()*torch.ones(S,yLen,L*J) \
            - ϕ(padAssetsF(Bf,side=1))

        #Euler Errors: equity then bond: THIS IS JUST E[MR]-1=0
        eqEuler = torch.sum(torch.abs(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])*(Pf+Δf_)/P,torch.tensor(probs),dims=([0],[0]))-1),-1)
        bondEuler = torch.prod(torch.abs(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])/Q,torch.tensor(probs),dims=([0],[0]))-1),-1)

        #Market Clearing Errors
        equityMCC = torch.abs(1-torch.sum(E,-1))
        bondMCC = torch.abs(torch.sum(B,-1))

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
        lr=1e-6
        return Adam(self.model.parameters(), lr=lr)

#make sure we're on GPU 
model = MODEL().to("cuda")

#This generates the training data
class CustDataSet(Dataset):
    def __init__(self,pretrain=False):
        model.to("cuda") #make sure we're on GPU
        model.eval() #turn off dropout
        
        #state variables: X
        #prediction variables: Y
        shist, zhist = SHOCKS() #shock path

        #Firms
        Yhist = F*zhist
        δhist = δ*zhist
        yhist = y*zhist

        #Network training data and predictions
        X = torch.zeros(T,input)
        Y = torch.zeros(T,output)
        X[0] = torch.concat([ebar.flatten(),bbar.flatten(),yhist[0].flatten(),Yhist[0,0],δhist[0,0]],0)
        Y[0] = model(X[0])
        for t in range(1,T):
            X[t] = torch.concat([Y[t-1,equity],Y[t-1,bond],yhist[t].flatten(),Yhist[t,0],δhist[t,0]],0)
            Y[t] = model(X[t])
        
        #lop off burn period
        X, Y = X[time], Y[time]
        
        #training inputs
        self.X = X.detach().clone()

        #training labels: if pretrain then detSS otherwise zeros
        if pretrain:
            self.Y = torch.ones(train,output)*torch.concat([ebar.flatten(),bbar.flatten(),pbar.flatten(),qbar.flatten()],0)
        else:
            self.Y = torch.zeros(train).float()

        model.train() #turn on dropout for training

    #mandatory to determine length of data
    def __len__(self):
        return len(self.Y)
    
    #return index batches
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]

ds = CustDataSet()