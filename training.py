#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3
#-------------------------------------------------------------------------------

from packages import *
from params import *
from nn import *

def pretrain_loop(epochs=100,lr=1e-6):
    
    #generate pretraining data: labels are detSS
    data = CustDataSet(pretrain=True) 
    
    for epoch in range(epochs):

        #load training data into DataLoader object for batching (ON GPU)
        train_loader = DataLoader(
            data,batch_size=50,generator=torch.Generator(device=device),
            shuffle=True,num_workers=0
        )

        #optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        #actual loop
        for batch, (X,y) in enumerate(train_loader):
            y_pred = model(X)
            lossval = loss(y_pred,y)

            optimizer.zero_grad()
            lossval.backward()
            optimizer.step()
        
        #after loop: what is loss?
        print(lossval.item())

def train_loop(epochs=100,lr=1e-8):
    
    for epoch in range(epochs):
        #generate pretraining data: labels are detSS
        data = CustDataSet(pretrain=False) 

        #load training data into DataLoader object for batching (ON GPU)
        train_loader = DataLoader(
            data,batch_size=32,generator=torch.Generator(device=device),
            shuffle=True,num_workers=0
        )

        #optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        #actual loop
        for batch, (X,y) in enumerate(train_loader):
            lossval = loss(model.losscalc(X),y)

            optimizer.zero_grad()
            lossval.backward()
            optimizer.step()
        
        #after loop: what is loss?
        print(lossval.item())