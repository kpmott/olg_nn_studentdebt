from packages import *
from parameters import PARAMS

#-------------------------------------------------------------------------------
#to solve for deterministic steady state (detSS) allocations, prices
#these are used as a starting point from which to train

class DET_SS_ALLOCATIONS():
    def __init__(self,g):
        self.params = PARAMS(g)

    def allocs(self):
        
        #everything to numpy
        y_np = self.params.y.cpu().numpy()
        debtPay_np = self.params.debtPay.cpu().numpy()
        τ_np = self.params.τ.cpu().numpy()
        δ_np = self.params.δ.cpu().numpy()

        #compute lifetime consumption based on equity holdings e0 and prices p0
        def c_eq(e0,p0):
            #p0 ∈ ℜ^{L}:      prices from birth to death 
            #e0 ∈ ℜ^{J,L-1}:  equity holdings from birth to (death-1)
            
            #vector of consumption in each period 
            cons = np.zeros((self.params.J,self.params.L))
            cons[:,0] = y_np[:,:,0] - p0[0]*e0[:,0] \
                - debtPay_np[:,:,0] - τ_np[:,:,0]
            cons[:,-1] = y_np[:,:,-1] + (δ_np+p0[-1])*e0[:,-1] \
                - debtPay_np[:,:,-1] - τ_np[:,:,-1]
            
            for i in range(1,self.params.L-1):
                cons[:,i] = y_np[:,:,i] + (δ_np+p0[i])*e0[:,i-1] \
                    - p0[i]*e0[:,i] - debtPay_np[:,:,i] - τ_np[:,:,i]
            
            return cons

        #equilibrium conditions
        def ss_eq(x):
            
            x = np.array(x)

            #equity holdings for 1:(L-1)
            e = np.reshape(x[:-1],(self.params.J,self.params.L-1))
            #price
            p = np.reshape(x[-1],(1))
            
            #consumption must be nonnegative
            c = c_eq(e,p*np.ones((self.params.L)))
            cons = np.maximum(1e-12*np.ones((self.params.J,self.params.L)),c)

            #Euler equations
            ssVec = np.zeros((self.params.J,self.params.L-1))
            for i in range(0,self.params.L-1):
                ssVec[:,i] = p*self.params.up(cons[:,i]) \
                    - self.params.β*(p+δ_np)*self.params.up(cons[:,i+1]) 
            #market clearing
            ssVec = ssVec.flatten()
            ssVec = np.concatenate([ssVec,np.array([1-np.sum(e)])])

            #in equilibrium all of these conditions should be zero: to solver 
            return ssVec
            
        #Guess equity is hump-shaped
        eguess = np.array(
            [
            norm.pdf(range(1,self.params.L),
            self.params.wp,self.params.wp) for j in range(self.params.J)
            ]
        )
        eguess /= np.sum(eguess)
        
        pguess = 3.25 + 2*floor(self.params.L/30)
        guess = np.append(eguess,pguess)

        #if the solver can't find detSS: quit
        if fsolve(ss_eq,guess,full_output=1,maxfev=int(10e8))[-2] != 1:
            print('Solution not found for detSS!')
            exit()
        else:
            #solution
            bar = torch.tensor(
                fsolve(ss_eq,guess,full_output=0,maxfev=int(10e8)))\
                    .float().to(self.params.device)
            ebar = bar[0:-1].reshape((1,self.params.J,self.params.L-1)) #equity
            bbar = ebar*0 #bond: 0
            pbar = bar[-1].reshape((1,1,1)) #equity price
            qbar = 1/((pbar+self.params.δ)/pbar) #bond price: equalize returns
            cbar = torch.tensor(
                c_eq(
                    torch.squeeze(ebar).cpu().numpy(),
                    pbar.flatten().cpu().numpy()*np.ones(self.params.L)
                )
            ).reshape((1,self.params.J,self.params.L))\
            .float().to(self.params.device) #consumption

        return ebar,bbar,pbar,qbar,cbar

    def Plots(self):
        ebar,_,_,_,cbar = self.allocs()

        lblsNums=['j='+str(j)+': ' for j in range(self.params.J)]
        lblsWords=['HS','Community College','College']
        lbls=[lblsNums[j]+lblsWords[j] for j in range(self.params.J)]
        figsize = (10,4)

        plt.figure(figsize=figsize);plt.plot(cbar.squeeze().t().cpu());\
        plt.legend(lbls);plt.title('detSS Consumption');plt.xlabel("i");\
        plt.xticks([i for i in range(self.params.L)]);\
        plt.savefig(self.params.plotPath+'.detSS_C.png');plt.clf()
        
        plt.figure(figsize=figsize);plt.plot(ebar.squeeze().t().cpu());\
        plt.legend(lbls);plt.title('detSS Equity Ownership');plt.xlabel("i");\
        plt.xticks([i for i in range(self.params.L-1)]);\
        plt.savefig(self.params.plotPath+'.detSS_E.png');plt.clf()

        #os.system("cp .detSS_C.png /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/")
        #os.system("mv /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/.detSS_C.png /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/detSS_C.png")
        #os.system("cp .detSS_E.png /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/")
        #os.system("mv /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/.detSS_E.png /home/kpmott/Dropbox/Apps/Overleaf/Dissertation/1_ApplicationStudent/detSS_E.png")