from packages import *

from governance import MODEL,forgivenesslist,bondprice,CustDataSet,annualize

rlist = []
rlistperiod = []

for g in range(9):
    model = MODEL()
    model.eval()
    model.load_state_dict(torch.load('./train/'+str(g)+'/.trained_model.pt'))
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
plt.plot(forgivenesslist,rlist)
#plt.plot(forgivenesslist,forgivenesslist,'--k')
plt.xlabel('Student Loan Cancelation Percentage');plt.ylabel('r: Private Borrowing Rate')
#plt.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.title('Private Borrowing Rate as a function of Student Loan Cancelation Percentage')
plt.savefig('cancelation.png');plt.clf();plt.close()

