from packages import *

class PARAMS():
    def __init__(self,g):
        rbarlist = [0.025+0.005*i for i in range(8)]
        devicelist = ["cuda:"+str(i) for i in range(8)]

        self.device = torch.device(
            devicelist[g*0] if torch.cuda.is_available() else "cpu"
        )
        self.rbar = rbarlist[g]
        
        #-----------------------------------------------------------------------
        #PARAMS 
        #EVERYTHING IS INDEXED [t,j,i]
        #Except when it needs to be a single vector

        #-----------------------------------------------------------------------
        #economic parameters

        #Lifespan of agents
        self.L = 4

        #Types of agents
        self.J = 3

        #Assets
        self.N = 2

        #working periods and retirement periods : endowment=0 in retirement
        self.wp = 3
        self.rp = self.L - self.wp

        #Time discount rate
        self.β = .995**(60/self.L)

        #Endowments: HK and debt for types j∈{0,1,2}
        #calibrated for relative incomes
        #https://educationdata.org/student-loan-debt-by-income-level
        self.hkEndow = (torch.ones((1,self.J,self.L))*\
            torch.tensor([[100],[123.6],[171.9]]).repeat(1,self.L)/100)\
            .to(self.device)

        #flags
        self.isWorker = torch.concat(
            [torch.ones((1,self.J,self.wp)),torch.zeros(1,self.J,self.rp)],
            -1).to(self.device)
        self.isRetired = torch.concat(
            [torch.zeros((1,self.J,self.wp)),torch.ones(1,self.J,self.rp)],
            -1).to(self.device)
        self.isEducated = torch.concat(
            [torch.zeros(1,1,self.L),torch.ones(1,self.J-1,self.L)],
            -2).to(self.device)

        self.isNotYoungest = torch.where(
            torch.tensor(
                [i%self.L for i in range(self.J*self.L)]
            ).float()==0,
            False,True
        )
        self.isNotOldest = torch.where(
            torch.tensor(
                [i%self.L for i in range(self.J*self.L)]
            ).float()==self.L-1,False,True
        )

        #-----------------------------------------------------------------------
        #stochastics 
        self.probs = [0.5, 0.5] #prob of each state
        self.S = len(self.probs)  #number of states
        self.ζ = 0.035 #aggregate shock size 
        self.shocks = torch.tensor([1-self.ζ,1+self.ζ]).to(self.device)

        #-----------------------------------------------------------------------
        #utility function 

        #Risk-aversion coeff
        self.γ = 3

        #-----------------------------------------------------------------------
        #production function
        ξ = 0.25

        def production():
            #{\bar h}^ξ
            barH = torch.sum(self.hkEndow*self.isEducated*self.isWorker)\
                /torch.sum(self.isEducated*self.isWorker)
            
            #h^(1-ξ)
            H = torch.sum(self.hkEndow*self.isWorker)

            return (barH**ξ*H**(1-ξ)).to(self.device)


        def wage():
            #h^-ξ
            barH = torch.sum(self.hkEndow*self.isEducated*self.isWorker)\
                /torch.sum(self.isEducated*self.isWorker)

            #{\bar h}^ξ
            H = torch.sum(self.hkEndow*self.isWorker)

            return (barH**ξ*H**(-ξ)*(1-ξ)).to(self.device)
            

        self.F = production().to(self.device)
        self.Fprime = wage().to(self.device)
        self.y = self.Fprime*self.hkEndow*self.isWorker
        self.δ = self.F - torch.sum(self.y)

        #Calibrate debt to match percent of income
        #https://educationdata.org/student-loan-debt-by-income-level
        self.debtEndow = (self.y[:,:,0]*torch.tensor([0,.44,.59])\
            .to(self.device))[:,:,None]

        #-----------------------------------------------------------------------
        #time path 

        self.T = 15000 #number of periods to simulate
        self.burn = 2500 #burn period: throw this away
        self.train = self.T - self.burn #the number of periods to train on 
        self.time = slice(self.burn,self.T,1) #training period slice

        #-----------------------------------------------------------------------
        #input

        #input dims
        #        assets_t  + incomes_t + totalresources_t  + div_t
        self.input = self.N*self.J*(self.L-1) + self.J*self.L + 1 + 1

        #slices to grab input
        self.typesState = slice(0,-2,1) #type-speciific state variables
        self.incomes = slice(
            2*self.J*(self.L-1),2*self.J*(self.L-1)+self.L*self.J,1
        )
        self.resources = slice(-2,-1,1)
        self.divs = slice(-1,self.input,1)
        self.aggState = slice(-2,self.input,1) 

        #-----------------------------------------------------------------------
        # output
        #        assets    + prices
        self.output = self.N*self.J*(self.L-1) + self.N

        #AGGREGATE VECTOR SLICES
        #all assets, all prices
        self.assets = slice(0,-2,1)
        self.equity = slice(0,self.J*(self.L-1),1)
        self.bond = slice(self.J*(self.L-1),2*self.J*(self.L-1),1)
        self.prices  = slice(-2,self.output,1)
        self.eqprice = slice(-2,-1,1)
        self.bondprice = slice(-1,self.output,1)

        #-----------------------------------------------------------------------
        self.savePath    = './train/'+str(g)
        self.plotPath = './plots/'+str(g)+'/'
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        if not os.path.exists(self.plotPath):
            os.makedirs(self.plotPath)

        #-----------------------------------------------------------------------
        #amortization function
        #student debt interest rate

        def amort(rbar):
            if rbar == 0:
                payment = 1/self.wp
            else:
                payment = rbar*(1+rbar)**self.wp/((1+rbar)**self.wp-1)
            return payment

        self.amortPay = amort(self.rbar)

        #debt payment per period
        self.debtPay = self.debtEndow*self.amortPay*self.isWorker

        #how much total tax revenue to raise
        taxRev = torch.sum(self.debtEndow[:,:,0]) - torch.sum(self.debtPay)

        #tax/transfer 
        τ = self.y[0,:,0][:,None]*torch.ones(1,self.J,self.L)
        τ /= torch.sum(τ)
        self.τ = τ*taxRev

    #---------------------------------------------------------------------------
    #utility
    def u(self,x):
        if self.γ == 1:
            return np.log(x)
        else:
            return (x**(1-self.γ))/(1-self.γ)

    #utility derivative
    def up(self,x):
        return x**-self.γ

    #inverse of utility derivative
    def upinv(self,x):
        if self.γ == 0:
            return x
        else:
            return x**(1/self.γ)
    
    #---------------------------------------------------------------------------
    #borrowing cost function
    def ϕ(self,b,λ=-0.25):
        return torch.where(torch.greater_equal(b,0.),torch.zeros(b.shape),λ*b)
    
    #---------------------------------------------------------------------------
    #Annualize rates
    #r in my model --> r in one year
    def annualize(self,r):
        return (1+r)**(self.L/60)-1

    #r in one year --> r in my model
    def periodize(self,r):
        return (1+r)**(60/self.L)-1

    #---------------------------------------------------------------------------
    #draw path of shocks 
    def SHOCKS(self):
        #shocks: {1,...,S=2}
        shocks = range(self.S)
        
        #Shock history:
        shist = torch.tensor(
            np.random.choice(shocks,self.T,self.probs),dtype=torch.int32)\
                .reshape((self.T,1,1)).to(self.device)
        zhist = shist*self.ζ*2-self.ζ + 1

        return shist, zhist

    #---------------------------------------------------------------------------
    #Tensor operations for later
    def padAssets(self,ASSETS,yLen,side=0):
        ASSETSJ = ASSETS.reshape(yLen,self.L-1,self.J)
        if side==0:
            ASSETSJpad = nn.functional.pad(ASSETSJ,(1,0))
        else:
            ASSETSJpad = nn.functional.pad(ASSETSJ,(0,1))
        return torch.flatten(ASSETSJpad,start_dim=-2).to(self.device)

    def padAssetsF(self,ASSETS,yLen,side=0):
        ASSETSJ = ASSETS.reshape(self.S,yLen,self.L-1,self.J)
        if side==0:
            ASSETSJpad = nn.functional.pad(ASSETSJ,(1,0))
        else:
            ASSETSJpad = nn.functional.pad(ASSETSJ,(0,1))
        return torch.flatten(ASSETSJpad,start_dim=-2).to(self.device)