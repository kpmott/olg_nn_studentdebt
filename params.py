from packages import *

#EVERYTHING IS INDEXED [t,i,j]
#Except when it needs to be a single vector

#--------------------------------------------------------------------------------
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
β = .985**(60/L)

#Endowments: HK and debt for types j∈{0,1,2}, calibrated for relative incomes
#https://educationdata.org/student-loan-debt-by-income-level
hkEndow = torch.ones((1,L,J))*torch.tensor([100,123.6,171.9])/100 

#flags
isWorker = torch.concat([torch.ones((1,wp,J)),torch.zeros(1,rp,J)],-2)
isRetired = torch.concat([torch.zeros((1,wp,J)),torch.ones(1,rp,J)],-2)
isEducated = torch.concat([torch.zeros(1,L,1),torch.ones(1,L,J-1)],-1)

#--------------------------------------------------------------------------------
#stochastics 
probs = [0.5, 0.5] #prob of each state
S = len(probs)  #number of states
ζ = 0.05 #aggregate shock size 
shocks = torch.tensor([1-ζ,1+ζ])

#--------------------------------------------------------------------------------
#utility function 

#Risk-aversion coeff
γ = 3.

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
    return x**(1/γ)

#--------------------------------------------------------------------------------
#production function
ξ = 0.25

def production():
    #{\bar h}^ξ
    barH = torch.sum(hkEndow*isEducated*isWorker)/torch.sum(isEducated*isWorker)
    
    #h^(1-ξ)
    H = torch.sum(hkEndow)

    return barH**ξ*H**(1-ξ)


def wage():
    #h^-ξ
    barH = torch.sum(hkEndow*isEducated*isWorker)/torch.sum(isEducated*isWorker)

    #{\bar h}^ξ
    H = torch.sum(hkEndow)

    return barH**ξ*H**(-ξ)*(1-ξ)
    

F = production()
Fprime = wage()
y = Fprime*hkEndow*isWorker
δ = F - torch.sum(y)

#Calibrate debt to match percent of income
#https://educationdata.org/student-loan-debt-by-income-level
debtEndow = y[:,0,:]*torch.tensor([0,.44,.59]).reshape((1,1,J))

#--------------------------------------------------------------------------------
#borrowing cost function
#borrowing cost parameter
λ = -0.025

def ϕ(b):
    return b*0#torch.where(torch.greater(b,0.),0.,λ*b)

#--------------------------------------------------------------------------------
#Annualize rates
#r in my model --> r in one year
def annualize(r):
    return (1+r)**(L/60)-1

#r in one year --> r in my model
def periodize(r):
    return (1+r)**(60/L)-1

#--------------------------------------------------------------------------------
#amortization function
#student debt interest rate

def amort(rbar=periodize(0.025)):
    if rbar == 0:
        payment = 1/wp
    else:
        payment = rbar*(1+rbar)**wp/((1+rbar)**wp-1)
    return payment

amortPay = amort()

#debt payment per period
debtPay = debtEndow*amortPay*isWorker

#how much total tax revenue to raise
taxRev = torch.sum(debtEndow[:,0,:]) - torch.sum(debtPay)

#tax/transfer 
τ = y*isWorker
τ /= torch.sum(τ)
τ *= taxRev

#--------------------------------------------------------------------------------
#time path 

T = 12500 #number of periods to simulate
burn = 5000 #burn period: throw this away
train = T - burn #the number of periods to train on 
time = slice(burn,T,1) #training period slice

#draw path of shocks 
def SHOCKS():
    #shocks: {1,...,S=2}
    shocks = range(S)
    
    #Shock history:
    shist = torch.tensor(np.random.choice(shocks,T,probs),dtype=torch.int32).reshape((T,1,1))
    zhist = shist*ζ*2-ζ + 1

    return shist, zhist

#--------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------
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

#This takes in output tensor (list) and returns output tensor (matrix: rows are types)
def typeOut(tensList):
    equityHoldingsJ = tensList[...,equity].reshape(J,L-1)
    bondholdingsJ = tensList[...,bond].reshape(J,L-1)
    pricesJ = tensList[...,prices].repeat(J,1)
    return torch.concat([equityHoldingsJ,bondholdingsJ,pricesJ],-1)