from packages import *

from forgiveness import MODEL,rbarlist,bondprice,CustDataSet,annualize

rlist = []
rlistperiod = []

for rbar in range(8):
    model = MODEL()
    model.eval()
    model.load_state_dict(torch.load('./train/'+str(rbar)+'/.trained_model.pt'))
    model.eval()
    DataSet = CustDataSet(model)
    x = DataSet.X
    y = model(x).detach()
    Q = y[...,bondprice]
    rlist.append(torch.mean(annualize(1/Q-1)).cpu().detach())
    rlistperiod.append(torch.mean(1/Q-1).cpu().detach())
    torch.cuda.empty_cache()
    del model, DataSet

plt.figure(figsize=(10,6))
plt.plot(rbarlist,rlist)
plt.plot(rbarlist,rbarlist,'--k')
plt.xlabel('rbar: Student Loan Rate');plt.ylabel('r: Private Borrowing Rate')
plt.title('Private Borrowing Rate as a function of Student Loan Rate')
plt.savefig('r_rbar.png');plt.clf();plt.close()

