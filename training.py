from packages import *
from packages import *
from parameters import PARAMS
from nn import MODEL
from detSS import DET_SS_ALLOCATIONS
from dataset import DATASET
        
class TRAIN():
    def __init__(self,g,saveTrain=True):
        self.params = PARAMS(g)
        self.dataset = DATASET(g)
        self.model = MODEL(g)
        self.saveTrain = saveTrain

    def train_loop(self,epochs=100,batchsize=32,lr=1e-8,losses=[]):
        tol = 1e-3

        for epoch in tqdm(range(epochs)):
            #generate pretraining data: labels are detSS
            data = self.dataset

            #load training data into DataLoader object for batching (ON GPU)
            train_loader = DataLoader(
                data,batch_size=batchsize,
                generator=torch.Generator(device=self.params.device),
                shuffle=True,num_workers=0
            )

            #optimizer
            optimizer = Adam(self.model.parameters(), lr=lr)

            #actual loop
            batchloss = []
            for batch, (X,y) in enumerate(train_loader):
                    
                optimizer.zero_grad()
                lossval = self.model.loss(self.model.losscalc(X),y)
                batchloss.append(lossval.item())
                lossval.backward()#create_graph=True)
                optimizer.step()
            
            epochloss = np.mean(batchloss)
            losses.append(epochloss)
            
            figsize = (10,4)
            plt.figure(figsize=figsize)
            plt.plot(losses)
            plt.yscale('log')
            plt.title("Losses: "+"{:.2e}".format(losses[-1]))
            plt.xlabel("Epoch")
            plt.savefig(self.params.plotPath+'.plot_losses.png')
            plt.clf()
            plt.close()

            if epochloss < tol:
                break
            
            torch.cuda.empty_cache()
        
        return losses
    
    def train(self):
        
        num_regimes = 3
        epochs = [500 for epoch in range(num_regimes)]
        batches = [2**(5+batch) for batch in range(num_regimes)]
        lrs = [1e-5*10**(-n) for n in range(num_regimes)]

        losses=[]
        for regime in range(num_regimes):
            losses = self.train_loop(
                    epochs=epochs[regime],
                    batchsize=batches[regime],
                    lr=lrs[regime],
                    losses=losses
            )

            if self.saveTrain:
                self.model.eval()
                torch.save(self.model.state_dict(), 
                    self.params.savePath+'/.trained_model_params.pt'
                )
                self.model.train()