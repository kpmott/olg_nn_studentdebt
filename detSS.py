from packages import *
from params import *

#-----------------------------------------------------------------------------------------------------------------
#to solve for deterministic steady state (detSS) allocations, prices
#these are used as a starting point from which to train

def detSS_allocs():
    
    #compute lifetime consumption based on equity holdings e0 and prices p0
    def c_eq(e0,p0):
        #p0 ∈ ℜ^L:      prices from birth to death 
        #e0 ∈ ℜ^{L-1}:  equity holdings from birth to (death-1)
        
        #vector of consumption in each period 
        cons = np.zeros(L)
        cons[0] = ω_scalar[0]-p0[0]*e0[0]
        cons[-1] = ω_scalar[-1]+(p0[-1]+δ_scalar)*e0[-1]
        for i in range(1,L-1):
            cons[i] = ω_scalar[i]+(p0[i]+δ_scalar)*e0[i-1]-p0[i]*e0[i]
        
        return cons

    #lifetime cash-on-hand
    def x_eq(e0,p0):
        #p0 ∈ ℜ^L:      prices from birth to death 
        #e0 ∈ ℜ^{L-1}:  equity holdings from birth to (death-1)
        
        #vector of consumption in each period 
        x = np.zeros(L)
        x[0] = ω_scalar[1]
        x[-1] = ω_scalar[-1]+(p0[-1]+δ_scalar)*e0[-1]
        for i in range(1,L-1):
            x[i] = ω_scalar[i]+(p0[i]+δ_scalar)*e0[i-1]
        
        return x

    #equilibrium conditions
    def ss_eq(x):
        #equity holdings for 1:(L-1)
        e = x[:-1]
        #price
        p = x[-1]
        
        #consumption must be nonnegative
        c = c_eq(e,p*np.ones(L))
        cons = np.maximum(1e-12*np.ones(L),c)

        #Euler equations
        ssVec = np.zeros(L)
        for i in range(0,L-1):
            ssVec[i] = p*up(cons[i]) - β*(p+δ_scalar)*up(cons[i+1])
        #market clearing
        ssVec[-1] = equitysupply - sum(e)

        #in equilibrium all of these conditions should be zero: pass to solver 
        return ssVec
        
    #Guess equity is hump-shaped
    eguess = norm.pdf(range(1,L),wp,.2*wp)
    eguess = [equitysupply*x/sum(eguess) for x in eguess]
    pguess = .85 + 2*floor(L/30)

    #if the solver can't find detSS: quit
    if fsolve(ss_eq,[*eguess,*[pguess]],full_output=1,maxfev=int(10e8))[-2] != 1:
        print('Solution not found for detSS!')
        exit()
    
    #solution
    bar = fsolve(ss_eq,[*eguess,*[pguess]],full_output=0,maxfev=int(10e8))
    ebar = bar[0:-1] #equity
    bbar = ebar*0 #bond: 0
    pbar = bar[-1] #equity price
    qbar = 1/((pbar+δ_scalar)/pbar) #bond price: equalize return to equity return 
    xbar = x_eq(ebar,pbar*np.ones(L)) #cash-on-hand
    cbar = c_eq(ebar,pbar*np.ones(L)) #consumption

    return ebar,bbar,pbar,qbar,xbar,cbar