#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3

from packages import *
from params import *

#-----------------------------------------------------------------------------------------------------------------
#to solve for deterministic steady state (detSS) allocations, prices
#these are used as a starting point from which to train

def detSS_allocs():
    
    y_np = y.cpu().numpy()
    debtEndow_np = debtEndow.cpu().numpy()
    τ_np = τ.cpu().numpy()
    δ_np = δ.cpu().numpy()
    ω_np = ω.cpu().numpy()

    #compute lifetime consumption based on equity holdings e0 and prices p0
    def c_eq(e0,p0):
        #p0 ∈ ℜ^{L}:      prices from birth to death 
        #e0 ∈ ℜ^{L-1,J}:  equity holdings from birth to (death-1)
        
        #vector of consumption in each period 
        cons = np.zeros((L,J))
        cons[0,:] = y_np[:,0,:] - p0[0]*e0[0,:] - a*debtEndow_np[:,0,:] - τ_np[:,0,:]
        cons[-1,:] = y_np[:,-1,:] + (δ_np+p0[-1])*e0[-1,:] - a*debtEndow_np[:,-1,:] - τ_np[:,-1,:]
        
        for i in range(1,L-1):
            cons[i,:] = y_np[:,i,:] + (δ_np+p0[i])*e0[i-1,:] - p0[i]*e0[i,:] - a*debtEndow_np[:,i,:] - τ_np[:,i,:]
        
        return cons

    #equilibrium conditions
    def ss_eq(x):
        
        x = np.array(x)

        #equity holdings for 1:(L-1)
        e = np.reshape(x[:-1],(L-1,J))
        #price
        p = np.reshape(x[-1],(1))
        
        #consumption must be nonnegative
        c = c_eq(e,p*np.ones((L)))
        cons = np.maximum(1e-12*np.ones((L,J)),c)

        #Euler equations
        ssVec = np.zeros((L-1,J))
        for i in range(0,L-1):
            ssVec[i,:] = p*up(cons[i,:]) - β*(p+δ_np)*up(cons[i+1,:])
        #market clearing
        ssVec = ssVec.flatten()
        ssVec = np.concatenate([ssVec,np.array([1-np.sum(e*ω_np[:,:-1,:])])])

        #in equilibrium all of these conditions should be zero: pass to solver 
        return ssVec

    def ss_eq_scalar(x):
        return np.sum(ss_eq(x))
        
    #Guess equity is hump-shaped
    eguess = np.repeat(norm.pdf(range(1,L),wp,wp-1),J)
    #eguess = [1/(J*(L-1)) for i in range(J*(L-1))]
    eguess = [x/sum(eguess)*J*L for x in eguess] 
    
    pguess = .05 + 2*floor(L/30)
    guess = [*eguess,*[pguess]]

    #if the solver can't find detSS: quit
    if fsolve(ss_eq,guess,full_output=1,maxfev=int(10e8))[-2] != 1:
        print('Solution not found for detSS!')
        exit()
    else:
        #solution
        bar = torch.tensor(fsolve(ss_eq,[*eguess,*[pguess]],full_output=0,maxfev=int(10e8))).float()
        ebar = bar[0:-1].reshape((1,L-1,J)) #equity
        bbar = ebar*0 #bond: 0
        pbar = bar[-1].reshape((1,1,1)) #equity price
        qbar = 1/((pbar+δ)/pbar) #bond price: equalize return to equity return 
        cbar = torch.tensor(c_eq(torch.squeeze(ebar).cpu().numpy(),pbar.flatten().cpu().numpy()*np.ones(L))).reshape((1,L,J)).float() #consumption

        return ebar,bbar,pbar,qbar,cbar

ebar,bbar,pbar,qbar,cbar = detSS_allocs()

lblsNums=['j='+str(j)+': ' for j in range(J)]
lblsWords=['HS','CommColl','Coll']
lbls=[lblsNums[j]+lblsWords[j] for j in range(J)]
plt.plot(torch.squeeze(cbar.cpu()));plt.legend(lbls);plt.title('detSS Consumption');plt.savefig('.detSS_C.png');plt.clf()
plt.plot(torch.squeeze(ω[:,:-1,:].cpu()*ebar.cpu()));plt.legend(lbls);plt.title('detSS Equity');plt.savefig('.detSS_E.png');plt.clf()