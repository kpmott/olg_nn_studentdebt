from torch import alpha_dropout
from packages import *

#EVERYTHING IS INDEXED [t,i,j]
#Except when it needs to be a single vector

#---------------------------------------------------------------------------------
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
β = 0.985**(60/L)

#Endowments: HK and debt for types j∈{0,1,2}
hkEndow = torch.ones((1,wp,J))*torch.tensor([1.00,1.30,1.60])*10
debtEndow = torch.ones((1,L,J))*torch.tensor([0,0,0])

#Population weights (start with equal)
ω = torch.ones(1,L,J)/(L*J)

#---------------------------------------------------------------------------------
#stochastics 
probs = [0.5, 0.5] #prob of each state
S = len(probs)  #number of states
ζtrue = 0.05 #aggregate shock size 

#-----------------------------------------------------------------------------------------------------------------
#utility function 

#Risk-aversion coeff
γ = 4.

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

#-----------------------------------------------------------------------------------------------------------------
#production function
ξ = 0.05

def production():
    #{\bar h}^ξ
    barH = torch.sum(ω[:,:wp,1:]*hkEndow[:,:wp,1:])/((J-1)*wp)
    
    #h^(1-ξ)
    H = torch.sum(ω[:,:wp,:]*hkEndow)

    return barH**ξ*H**(1-ξ)


def production_deriv():
    #h^-ξ
    barH = torch.sum(ω[:,:wp,1:]*hkEndow[:,:wp,1:])/((J-1)*wp)

    #{\bar h}^ξ
    H = torch.sum(ω[:,:wp,:]*hkEndow)

    return barH**ξ*H**(-ξ)*(1-ξ)*ω[:,:wp,:] #+ ξ*barH**(ξ-1)*H**(1-ξ)*nn.functional.pad(ω[:,:wp,1:],(1,0,0,0))/((J-1)*wp)
    

F = production()
Fprime = production_deriv()
y = torch.concat([Fprime*hkEndow,torch.zeros((1,rp,J))],-2)
δ = F - torch.sum(y)

#-----------------------------------------------------------------------------------------------------------------
#borrowing cost function
#borrowing cost parameter
λ = -0.025

def ϕ(b):
    return torch.where(torch.greater(b,0.),0.,λ*b)

#-----------------------------------------------------------------------------------------------------------------
#amortization function
#student debt interest rate

def amort(rbar=0.025):
    #d = torch.tensor(debtEndow,dtype=torch.float32)
    payment = rbar*(1+rbar)**wp/((1+rbar)**wp-1)
    return payment#*d

a = amort()
taxRev = torch.sum(debtEndow)*(1-a)
τ = taxRev*ω

#-----------------------------------------------------------------------------------------------------------------
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
    shist = torch.tensor(np.random.choice(shocks,T,probs),dtype=torch.int32)
    zhist = shist*ζtrue*2-ζtrue + 1

    return shist, zhist

#-----------------------------------------------------------------------------------------------------------------
#input

#input dims
#        assets_t  + incomes_t + totalresources_t  + div_t
input = N*J*(L-1) + J*L         + 1                 + 1

#slices to grab input
typesState = slice(0,-2,1) #type-speciific state variables
aggState = slice(-2,input,1) #aggregate quantities state variables

#This takes in input tensor (list) and returns input tensor (matrix: rows are types)
def typeIn(tensList):
    assetsIncomesJ = tensList[typesState].reshape(J,N*(L-1)+L) #assets and incomes by type j
    stateContingent = tensList[aggState].repeat(J,1) #state-contingent total resources and dividend
    return torch.concat([assetsIncomesJ,stateContingent],-1)

#-----------------------------------------------------------------------------------------------------------------
# output
#        assets    + prices
output = N*J*(L-1) + N

#AGGREGATE VECTOR SLICES
#all assets, all prices
assetsAll = slice(0,-2,1)
pricesAll  = slice(-2,output,1)

#WITHIN-TYPE SLICES
equity  = slice(0     ,L-1    ,1)
bond    = slice(L-1   ,2*L-2  ,1)
price   = slice(2*L-2 ,2*L-1  ,1)
ir      = slice(2*L-1 ,2*L    ,1)

#This takes in output tensor (list) and returns output tensor (matrix: rows are types)
def typeOut(tensList):
    assetsJ = tensList[assetsAll].reshape(J,N*(L-1))
    prices = tensList[pricesAll].repeat(J,1)
    return torch.concat([assetsJ,prices],-1)

#This takes in output tensor (matrix) and returns output tensor (list)
def vecOut(tensMat):
    assets = tensMat[...,:-2].flatten()
    prices = tensMat[-1,-2:]
    return torch.concat([assets,prices],0)