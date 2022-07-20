#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3
#-------------------------------------------------------------------------------

from packages import *
from params import *
from nn import *

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
            y_pred = model(X)
            lossval = loss(y_pred,y)
            batchloss.append(lossval.item())

            optimizer.zero_grad()
            lossval.backward()
            optimizer.step()

        losses.append(np.mean(batchloss))
        
        if epoch%20 ==0:
            plt.plot(losses);plt.yscale('log');\
                plt.title("Pre-Train Losses: "+"{:.2e}".format(losses[-1]));\
                plt.xlabel("Epoch");plt.savefig('.plot_prelosses.png');\
                plt.clf()

    return losses
        
def train_loop(epochs=100,batchsize=32,lr=1e-8,losses=[]):
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
            lossval = loss(model.losscalc(X),y)
            batchloss.append(lossval.item())

            optimizer.zero_grad()
            lossval.backward()
            optimizer.step()
        
        losses.append(np.mean(batchloss))
        
        if epoch%1==0:
            plt.plot(losses);plt.yscale('log');\
                plt.title("Losses: "+"{:.2e}".format(losses[-1]));\
                plt.xlabel("Epoch");plt.savefig('.plot_losses.png');\
                plt.clf()
        
    return losses