from packages import *
from parameters import PARAMS
from nn import MODEL
from detSS import DET_SS_ALLOCATIONS

#This generates the training data
class DATASET(Dataset):
    def __init__(self,g):
        model = MODEL(g)
        params = PARAMS(g)
        ebar,bbar,_,_,_ = DET_SS_ALLOCATIONS(g).allocs()

        model.eval() #turn off dropout
        
        #ignore gradients: faster
        with torch.no_grad():
            
            #state variables: X
            #prediction variables: Y
            _, zhist = params.SHOCKS() #shock path

            #Firms
            Yhist = params.F*zhist
            δhist = params.δ*zhist
            yhist = params.y*zhist

            #Network training data and predictions
            X = torch.zeros(params.T,params.input)\
                .to(params.device)
            Y = torch.zeros(params.T,params.output)\
                .to(params.device)
            X[0] = torch.concat(
                [ebar.flatten(),bbar.flatten(),
                yhist[0].flatten(),Yhist[0,0],δhist[0,0]],
                0
            )
            Y[0] = model(X[0])
            for t in range(1,params.T):
                X[t] = torch.concat(
                    [Y[t-1,params.equity],Y[t-1,params.bond],
                    yhist[t].flatten(),Yhist[t,0],δhist[t,0]],
                    0
                )
                Y[t] = model(X[t])
        
        #training inputs
        self.X = X[params.time]

        #training labels: zeros
        self.Y = torch.zeros(params.train).float()\
            .to(params.device).detach().clone()

        model.train() #turn on dropout for training

    #mandatory to determine length of data
    def __len__(self):
        return len(self.Y)
    
    #return index batches
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]