from packages import *

#---------------------------------------------------------------------------------
#economic parameters

#Lifespan of agents
L = 60

#working periods and retirement periods : endowment=0 in retirement
wp = int(L*2/3)
rp = L - wp

#Time discount rate
β = 1.#0.995**(60/L)

divshare = .33 #dividend share of total income 

#total resources
wbar = 1

#endowment share of total resources
ωGuess = norm.pdf(np.linspace(1,wp,wp),0.8*wp,L*0.25)
ωGuess = (1-divshare)*ωGuess/np.sum(ωGuess)

#share of total resources: [divident, working endowment, retirement endowment=0]
if L == 3:
    ls = np.array([1/3, 1/4, 5/12, 0])
else:
    ls = wbar*np.array([*[divshare], *ωGuess, *np.zeros(rp)])

#---------------------------------------------------------------------------------
#stochastics 
probs = [0.5, 0.5] #prob of each state
S = len(probs)  #number of states
ζtrue = 0.05 #aggregate shock size 
wvec = np.array([wbar*(1 - ζtrue), wbar*(1+ + ζtrue)])     #total resources in each state
δvec = np.multiply(ls[0],wvec)/wbar          #dividend in each state
ωvec = [ls[1:]*w/wbar for w in wvec]         #endowment process in each state

#convert endowment and dividend to tensors for later
ω = torch.tensor(np.array(ωvec))
δ = torch.reshape(torch.tensor(δvec),(S,1))

#mean-center: deterministic steady-state values
δ_scalar = ls[0]
ω_scalar = ls[1:]

#net supply of assets
equitysupply = 1
bondsupply = 0

#-----------------------------------------------------------------------------------------------------------------
#utility function 

#Risk-aversion coeff
γ = 2.

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
    shist = np.random.choice(shocks,T,probs)

    #History: endowments, dividends, resources
    Ωhist = [ωvec[t] for t in shist] #endowments
    Δhist = δvec[shist] #dividends
    whist = wvec[shist] #total resources 

    #convert to tensors now for easier operations later
    Ω = torch.tensor(np.array(Ωhist))
    Δ = torch.reshape(torch.tensor(Δhist),(T,1))

    return shist, whist, Ωhist, Δhist, Ω, Δ

#-----------------------------------------------------------------------------------------------------------------
#input/output size information and slices

"""
input   = [(e_i^{t-1})_i,(b_i^{t-1})_i,w^t,(ω_i^t)_i,δ^t]
          [lagged asset holdings,state-contingent total resources,state-contingent endowment,state-contingent dividend]
output  = [((e_i^{t})_{i=1}^{L-1},(b_i^{t})_{i=1}^{L-1},p^t,q^t]   ∈ ℜ^{2L}
          [asset holdings, prices]  
"""
#input/output dims
#        assets     + resources     + endowments    + div
input = 2*(L-1)     + 1             + L             + 1

#        assets     + prices
output = 2*(L-1)    + 2

#slices to grab output 
equity =    slice(0     ,L-1    ,1)
bond =      slice(L-1   ,2*L-2  ,1)
price =     slice(2*L-2 ,2*L-1  ,1)
ir =        slice(2*L-1 ,2*L    ,1)

#slices to grab input
endow = slice(2*L-1,    3*L-1,  1)
div =   slice(3*L-1,    3*L,    1)