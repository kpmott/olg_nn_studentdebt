#!/home/kpmott/Git/olg_nn_studentdebt/pyt_olg/bin/python3
#------------------------------------------------------------------------------------------------------

from packages import *
from params import *
from nn import *

#------------------------------------------------------------------------------------------------------
#STEP 1: PRETRAINING

#generate pretraining data: labels are detSS
data = CustDataSet(pretrain=True) 

#load training data into DataLoader object for batching (ON GPU)
train_loader = DataLoader(data,batch_size=50,generator=torch.Generator(device="cuda"),shuffle=True,num_workers=0)

#define and fit the trainer
trainer = pl.Trainer(max_epochs=200, accelerator="gpu",logger=False,enable_checkpointing=False)
trainer.fit(model=model,train_dataloaders=train_loader)

#------------------------------------------------------------------------------------------------------
#STEP 2: MAIN TRAINING

#maximum iterations
iters = 15000

#list of losses to track over training cycle
losshist = np.zeros((iters))

#machine tolerance: end when loss is below this
ϵ = 1e-3

#training loop: Draw data from unique path of shocks. Train with batch_size=__ for one epoch. Then repeat
for thyme in tqdm(range(iters)):
    
    #draw data and load (ON GPU) into DataLoader for batching
    data = CustDataSet()
    train_loader = DataLoader(data,batch_size=32,generator=torch.Generator(device="cuda"),shuffle=True)
    
    #define trainer
    trainer = pl.Trainer(max_epochs=1, accelerator="gpu",logger=False,enable_checkpointing=False,enable_model_summary=False)
    trainer.fit(model=model,train_dataloaders=train_loader)
    
    #check loss, plot
    # losses = model.losscalc(data.X)
    # lossrun = loss(losses,losses*0).cpu().detach().numpy()
    # losshist[thyme] = lossrun
    # plt.plot(losshist[:thyme+1]);plt.yscale('log');plt.savefig('plot_losses.png');plt.clf()
    # if lossrun < ϵ:
    #     print("Convergence in "+str(thyme)+" steps.")
    #     break