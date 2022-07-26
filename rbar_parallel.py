#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3

from packages import *

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-G", "--gpu", type=int, help="which gpu in [0,...,8]", default=0
)
parser.parse_args()
args = parser.parse_args()
config = vars(args)
g = config['gpu']

rbarlist = [0.025+0.005*i for i in range(8)]
devicelist = ["cuda:"+str(i) for i in range(8)]

device = torch.device(g if torch.cuda.is_available() else "cpu")
rbar = rbarlist[g]

#-------------------------------------------------------------------------------
#PARAMS 
#EVERYTHING IS INDEXED [t,j,i]
#Except when it needs to be a single vector

#-------------------------------------------------------------------------------
#economic parameters

#Lifespan of agents
L = 4

#Types of agents
J = 3

#Assets
N = 2

#working periods and retirement periods : endowment=0 in retirement
wp = 3
rp = L - wp

#Time discount rate
β = .995**(60/L)

#Endowments: HK and debt for types j∈{0,1,2}, calibrated for relative incomes
#https://educationdata.org/student-loan-debt-by-income-level
hkEndow = \
    (torch.ones((1,J,L))*torch.tensor([[100],[123.6],[171.9]]).repeat(1,L)/100)\
    .to(device)

#flags
isWorker = torch.concat(
    [torch.ones((1,J,wp)),torch.zeros(1,J,rp)],
    -1).to(device)
isRetired = torch.concat(
    [torch.zeros((1,J,wp)),torch.ones(1,J,rp)],
    -1).to(device)
isEducated = torch.concat(
    [torch.zeros(1,1,L),torch.ones(1,J-1,L)],
    -2).to(device)

isNotYoungest = torch.where(
    torch.tensor([i%L for i in range(J*L)]).float()==0,False,True
)
isNotOldest = torch.where(
    torch.tensor([i%L for i in range(J*L)]).float()==L-1,False,True
)

#-------------------------------------------------------------------------------
#stochastics 
probs = [0.5, 0.5] #prob of each state
S = len(probs)  #number of states
ζ = 0.035 #aggregate shock size 
shocks = torch.tensor([1-ζ,1+ζ]).to(device)

#-------------------------------------------------------------------------------
#utility function 

#Risk-aversion coeff
γ = 3

#utility
def u(x):
    if γ == 1:
        return np.log(x)
    else:
        return (x**(1-γ))/(1-γ)

#utility derivative
def up(x):
    return x**-γ

#inverse of utility derivative
def upinv(x):
    if γ == 0:
        return x
    else:
        return x**(1/γ)

#-------------------------------------------------------------------------------
#production function
ξ = 0.25

def production():
    #{\bar h}^ξ
    barH = torch.sum(hkEndow*isEducated*isWorker)/torch.sum(isEducated*isWorker)
    
    #h^(1-ξ)
    H = torch.sum(hkEndow*isWorker)

    return (barH**ξ*H**(1-ξ)).to(device)


def wage():
    #h^-ξ
    barH = torch.sum(hkEndow*isEducated*isWorker)/torch.sum(isEducated*isWorker)

    #{\bar h}^ξ
    H = torch.sum(hkEndow*isWorker)

    return (barH**ξ*H**(-ξ)*(1-ξ)).to(device)
    

F = production().to(device)
Fprime = wage().to(device)
y = Fprime*hkEndow*isWorker
δ = F - torch.sum(y)

#Calibrate debt to match percent of income
#https://educationdata.org/student-loan-debt-by-income-level
debtEndow = (y[:,:,0]*torch.tensor([0,.44,.59]).to(device))[:,:,None]

#-------------------------------------------------------------------------------
#borrowing cost function
#borrowing cost parameter
λ = -0.25

def ϕ(b):
    return torch.where(torch.greater_equal(b,0.),torch.zeros(b.shape),λ*b)

#-------------------------------------------------------------------------------
#Annualize rates
#r in my model --> r in one year
def annualize(r):
    return (1+r)**(L/60)-1

#r in one year --> r in my model
def periodize(r):
    return (1+r)**(60/L)-1

#-------------------------------------------------------------------------------
#time path 

T = 15000 #number of periods to simulate
burn = 2500 #burn period: throw this away
train = T - burn #the number of periods to train on 
time = slice(burn,T,1) #training period slice

#draw path of shocks 
def SHOCKS():
    #shocks: {1,...,S=2}
    shocks = range(S)
    
    #Shock history:
    shist = torch.tensor(
        np.random.choice(shocks,T,probs),dtype=torch.int32).reshape((T,1,1))\
        .to(device)
    zhist = shist*ζ*2-ζ + 1

    return shist, zhist

#-------------------------------------------------------------------------------
#input

#input dims
#        assets_t  + incomes_t + totalresources_t  + div_t
input = N*J*(L-1) + J*L         + 1                 + 1

#slices to grab input
typesState = slice(0,-2,1) #type-speciific state variables
incomes = slice(2*J*(L-1),2*J*(L-1)+L*J,1)
resources = slice(-2,-1,1)
divs = slice(-1,input,1)
aggState = slice(-2,input,1) #aggregate quantities state variables

#-------------------------------------------------------------------------------
# output
#        assets    + prices
output = N*J*(L-1) + N

#AGGREGATE VECTOR SLICES
#all assets, all prices
assets = slice(0,-2,1)
equity = slice(0,J*(L-1),1)
bond = slice(J*(L-1),2*J*(L-1),1)
prices  = slice(-2,output,1)
eqprice = slice(-2,-1,1)
bondprice = slice(-1,output,1)

#WITHIN-TYPE SLICES
equityJ  = slice(0     ,J*(L-1)    ,1)
bondJ    = slice(L-1   ,2*L-2  ,1)
priceJ   = slice(2*L-2 ,2*L-1  ,1)
irJ      = slice(2*L-1 ,2*L    ,1)

#output tensor (list) → output tensor (matrix: rows are types)
def typeOut(tensList):
    equityHoldingsJ = tensList[...,equity].reshape(J,L-1)
    bondholdingsJ = tensList[...,bond].reshape(J,L-1)
    pricesJ = tensList[...,prices].repeat(J,1)
    return torch.concat([equityHoldingsJ,bondholdingsJ,pricesJ],-1).to(device)

#-------------------------------------------------------------------------------
#Tensor operations for later
def padAssets(ASSETS,yLen,side=0):
    ASSETSJ = ASSETS.reshape(yLen,L-1,J)
    if side==0:
        ASSETSJpad = nn.functional.pad(ASSETSJ,(1,0))
    else:
        ASSETSJpad = nn.functional.pad(ASSETSJ,(0,1))
    return torch.flatten(ASSETSJpad,start_dim=-2).to(device)

def padAssetsF(ASSETS,yLen,side=0):
    ASSETSJ = ASSETS.reshape(S,yLen,L-1,J)
    if side==0:
        ASSETSJpad = nn.functional.pad(ASSETSJ,(1,0))
    else:
        ASSETSJpad = nn.functional.pad(ASSETSJ,(0,1))
    return torch.flatten(ASSETSJpad,start_dim=-2).to(device)

#-------------------------------------------------------------------------------
savePrePath = './.pretrained_model_params.pt'
savePath = './.trained_model_params.pt'
plotPath = './plots/'
if not os.path.exists(plotPath):
    os.makedirs(plotPath)

#-------------------------------------------------------------------------------
#amortization function
#student debt interest rate

def amort(rbar):
    if rbar == 0:
        payment = 1/wp
    else:
        payment = rbar*(1+rbar)**wp/((1+rbar)**wp-1)
    return payment

amortPay = amort(rbar)

#debt payment per period
debtPay = debtEndow*amortPay*isWorker

#how much total tax revenue to raise
taxRev = torch.sum(debtEndow[:,:,0]) - torch.sum(debtPay)

#tax/transfer 
τ = y[0,:,0][:,None]*torch.ones(1,J,L)
τ /= torch.sum(τ)
τ *= taxRev

#-------------------------------------------------------------------------------
#to solve for deterministic steady state (detSS) allocations, prices
#these are used as a starting point from which to train

def detSS_allocs():
    
    #everything to numpy
    y_np = y.cpu().numpy()
    debtPay_np = debtPay.cpu().numpy()
    τ_np = τ.cpu().numpy()
    δ_np = δ.cpu().numpy()

    #compute lifetime consumption based on equity holdings e0 and prices p0
    def c_eq(e0,p0):
        #p0 ∈ ℜ^{L}:      prices from birth to death 
        #e0 ∈ ℜ^{J,L-1}:  equity holdings from birth to (death-1)
        
        #vector of consumption in each period 
        cons = np.zeros((J,L))
        cons[:,0] = y_np[:,:,0] - p0[0]*e0[:,0] \
            - debtPay_np[:,:,0] - τ_np[:,:,0]
        cons[:,-1] = y_np[:,:,-1] + (δ_np+p0[-1])*e0[:,-1] \
            - debtPay_np[:,:,-1] - τ_np[:,:,-1]
        
        for i in range(1,L-1):
            cons[:,i] = y_np[:,:,i] + (δ_np+p0[i])*e0[:,i-1] \
                - p0[i]*e0[:,i] - debtPay_np[:,:,i] - τ_np[:,:,i]
        
        return cons

    #equilibrium conditions
    def ss_eq(x):
        
        x = np.array(x)

        #equity holdings for 1:(L-1)
        e = np.reshape(x[:-1],(J,L-1))
        #price
        p = np.reshape(x[-1],(1))
        
        #consumption must be nonnegative
        c = c_eq(e,p*np.ones((L)))
        cons = np.maximum(1e-12*np.ones((J,L)),c)

        #Euler equations
        ssVec = np.zeros((J,L-1))
        for i in range(0,L-1):
            ssVec[:,i] = p*up(cons[:,i]) - β*(p+δ_np)*up(cons[:,i+1]) 
        #market clearing
        ssVec = ssVec.flatten()
        ssVec = np.concatenate([ssVec,np.array([1-np.sum(e)])])

        #in equilibrium all of these conditions should be zero: pass to solver 
        return ssVec
        
    #Guess equity is hump-shaped
    eguess = np.array([norm.pdf(range(1,L),wp,wp) for j in range(J)])
    eguess /= np.sum(eguess)
    
    pguess = 3.25 + 2*floor(L/30)
    guess = np.append(eguess,pguess)

    #if the solver can't find detSS: quit
    if fsolve(ss_eq,guess,full_output=1,maxfev=int(10e8))[-2] != 1:
        print('Solution not found for detSS!')
        exit()
    else:
        #solution
        bar = torch.tensor(
            fsolve(ss_eq,guess,full_output=0,maxfev=int(10e8)))\
                .float().to(device)
        ebar = bar[0:-1].reshape((1,J,L-1)) #equity
        bbar = ebar*0 #bond: 0
        pbar = bar[-1].reshape((1,1,1)) #equity price
        qbar = 1/((pbar+δ)/pbar) #bond price: equalize return to equity return 
        cbar = torch.tensor(
            c_eq(
                torch.squeeze(ebar).cpu().numpy(),
                pbar.flatten().cpu().numpy()*np.ones(L))
            ).reshape((1,J,L)).float().to(device) #consumption

    return ebar,bbar,pbar,qbar,cbar

def Plots():
    ebar,bbar,pbar,qbar,cbar = detSS_allocs()

    lblsNums=['j='+str(j)+': ' for j in range(J)]
    lblsWords=['HS','Community College','College']
    lbls=[lblsNums[j]+lblsWords[j] for j in range(J)]
    figsize = (10,4)

    plt.figure(figsize=figsize);plt.plot(cbar.squeeze().t().cpu());\
    plt.legend(lbls);plt.title('detSS Consumption');plt.xlabel("i");\
    plt.xticks([i for i in range(L)]);\
    plt.savefig(plotPath+'.detSS_C.png');plt.clf()
    
    plt.figure(figsize=figsize);plt.plot(ebar.squeeze().t().cpu());\
    plt.legend(lbls);plt.title('detSS Equity Ownership');plt.xlabel("i");\
    plt.xticks([i for i in range(L-1)]);\
    plt.savefig(plotPath+'.detSS_E.png');plt.clf()

    #os.system("cp .detSS_C.png /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/")
    #os.system("mv /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/.detSS_C.png /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/detSS_C.png")
    #os.system("cp .detSS_E.png /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/")
    #os.system("mv /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/.detSS_E.png /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/detSS_E.png")

Plots()

#-------------------------------------------------------------------------------
"""
#this module defines - the neural network (with my custom loss function)
#                    - training data generation      
"""

#for use later: detSS allocs and prices
ebar,bbar,pbar,qbar,cbar = detSS_allocs()

#loss function for comparing "output" to labels
loss = nn.L1Loss()

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
        dp=0.175 #dropout parameter
        self.model = nn.Sequential(
            nn.Linear(in_features=sizes[0],out_features=sizes[1]),
            nn.ReLU(),nn.Dropout(p=0.),
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
    def losscalc(self,x,full_loss=False):
        
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
        eqEuler = torch.mean(torch.abs(
            β*torch.tensordot(up(Cf[...,isNotYoungest])*(Pf+Δf_)
            ,torch.tensor(probs),dims=([0],[0]))/(up(C[...,isNotOldest])*P) - 1.
            ),-1
        )[:,None]

        bondEuler = torch.mean(torch.abs(
            β*torch.tensordot(up(Cf[...,isNotYoungest])
            ,torch.tensor(probs),dims=([0],[0]))/(up(C[...,isNotOldest])*Q) - 1.
            ),-1
        )[:,None]

        #Market Clearing Errors
        equityMCC = torch.abs(1-torch.sum(E,-1))[:,None]
        bondMCC   = torch.abs(torch.sum(B,-1))[:,None]

        #so in each period, the error is the sum of above
        #EULERS + MCCs + consumption penalty 
        loss_vec = torch.concat([eqEuler,bondEuler,equityMCC,bondMCC,cpen],-1)

        #set model back to training (dropout back on)
        self.model.train()

        if full_loss:
            return loss_vec
        else:
            p=1.
            return torch.sum(loss_vec**p,-1)**(1/p)
        #return torch.log(torch.sum(loss_vec,-1)+1)

    
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
                0
            )
            Y[0] = model(X[0])
            for t in range(1,T):
                X[t] = torch.concat(
                    [Y[t-1,equity],Y[t-1,bond],
                    yhist[t].flatten(),Yhist[t,0],δhist[t,0]],
                    0
                )
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
                ).clone()
        else:
            self.Y = torch.zeros(train).float().to(device).detach().clone()

        model.train() #turn on dropout for training

    #mandatory to determine length of data
    def __len__(self):
        return len(self.Y)
    
    #return index batches
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]

#-------------------------------------------------------------------------------
def pretrain_loop(epochs=100,batchsize=50,lr=1e-6,losses=[]):
    
    #generate pretraining data: labels are detSS
    data = CustDataSet(pretrain=True) 
    
    for epoch in tqdm(range(epochs)):

        #load training data into DataLoader object for batching (ON GPU)
        train_loader = DataLoader(
            data,batch_size=batchsize,generator=torch.Generator(device=device),
            shuffle=True,num_workers=0
        )

        #optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        #actual loop
        batchloss = []
        for batch, (X,y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X)
            lossval = loss(y_pred,y)
            batchloss.append(lossval.item())

            lossval.backward()
            optimizer.step()

        losses.append(np.mean(batchloss))
        
        figsize = (10,4)
        if epoch%20 ==0:
            plt.figure(figsize=figsize)
            plt.plot(losses);plt.yscale('log');\
            plt.title("Pre-Train Losses: "+"{:.2e}".format(losses[-1]));\
            plt.xlabel("Epoch");plt.savefig(plotPath+'.plot_prelosses.png');\
            plt.clf()
            plt.close()

    return losses
        
def train_loop(epochs=100,batchsize=32,lr=1e-8,losses=[]):
    tol = 1e-3

    for epoch in tqdm(range(epochs)):
        #generate pretraining data: labels are detSS
        data = CustDataSet(pretrain=False) 

        #load training data into DataLoader object for batching (ON GPU)
        train_loader = DataLoader(
            data,batch_size=batchsize,generator=torch.Generator(device=device),
            shuffle=True,num_workers=0
        )

        #optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        #actual loop
        batchloss = []
        for batch, (X,y) in enumerate(train_loader):
                
            optimizer.zero_grad()
            lossval = loss(model.losscalc(X),y)
            batchloss.append(lossval.item())
            lossval.backward()#create_graph=True)
            optimizer.step()
        
        epochloss = np.mean(batchloss)
        losses.append(epochloss)
        
        figsize = (10,4)

        plt.figure(figsize=figsize)
        plt.plot(losses);plt.yscale('log');\
        plt.title("Losses: "+"{:.2e}".format(losses[-1]));\
        plt.xlabel("Epoch");plt.savefig(plotPath+'.plot_losses.png');\
        plt.clf()
        plt.close()

        if epochloss < tol:
            break
        
        torch.cuda.empty_cache()
    
    return losses

#-------------------------------------------------------------------------------
savePretrain = False
loadPretrain = False
saveTrain = True
#-------------------------------------------------------------------------------
#STEP 1: PRETRAINING

if savePretrain:
    losses = pretrain_loop(epochs=250,batchsize=50,lr=1e-4)
    losses = pretrain_loop(epochs=250,batchsize=50,lr=1e-5,losses=losses)
    losses = pretrain_loop(epochs=250,batchsize=100,lr=1e-6,losses=losses)
    losses = pretrain_loop(epochs=250,batchsize=200,lr=1e-6,losses=losses)

    model.eval()
    torch.save(model.state_dict(), savePrePath)
    model.train()

#-------------------------------------------------------------------------------
#STEP 2: MAIN TRAINING

if loadPretrain:
    model.load_state_dict(torch.load(savePrePath))

losses = train_loop(epochs=500,batchsize=32,lr=1e-5)
if saveTrain:
    model.eval()
    torch.save(model.state_dict(), savePath)
    model.train()

losses = train_loop(epochs=500,batchsize=64,lr=1e-6,losses=losses)
if saveTrain:
    model.eval()
    torch.save(model.state_dict(), savePath)
    model.train()

losses = train_loop(epochs=500,batchsize=100,lr=1e-7,losses=losses)
if saveTrain:
    model.eval()
    torch.save(model.state_dict(), savePath)
    model.train()

#-------------------------------------------------------------------------------
model.eval()
model.load_state_dict(torch.load(savePath))

#-------------------------------------------------------------------------------
#Data
data = CustDataSet()
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