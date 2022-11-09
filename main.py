import random
import os
import argparse
import matplotlib.pyplot as plt
from PathExpSurv import PathExpSurv
from Survival import  neg_par_log_likelihood, c_index
from utils import pathway2bool_adjusted, pathway2bool
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='PathExpSurv')
parser.add_argument('--dataset',type=str,default='BRCA',help='available datasets: BRCA | THCA | LGG')
parser.add_argument('--model',type=str,default='pathexpsurv',help='available models: ori | adj | full | pathexpsurv')
parser.add_argument('--total_fold',type=int, default=10, help='num of fold')
parser.add_argument('--lr',type=float, default=0.05, help='learning rate')
parser.add_argument('--num_epochs',type=int, default=200, help='total number of epochs')
parser.add_argument('--lambda_',type=float, default=1, help='penalty weight 1')
parser.add_argument('--mu',type=float, default=1, help='penalty weight 2')
args=parser.parse_args()

from sklearn.model_selection import KFold
def Split_Sets_10_Fold(total_fold, data):   
    train_index = []
    test_index = []
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=True)
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)
    return train_index, test_index

if __name__=='__main__':
    dtype = torch.FloatTensor

    gpu_id=0
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')

    data_all=pd.read_csv("Dataset/{}/{}_pro_select.csv".format(args.dataset,args.dataset))

    genes = np.array(data_all.columns[:-2])



    if args.model=='adj':
        # adjusted pathway
        path="Results/{}/adjusted_pathway.csv".format(args.dataset)
        pathway_mask=pathway2bool_adjusted(path,genes)
    elif args.model=='ori' or args.model=='pathexpsurv':
        #original pathway
        path="Dataset/{}/{}_kegg.csv".format(args.dataset, args.dataset)
        pathway_mask=pathway2bool(path,genes)
    elif args.model=='full':
        #fully connected
        path="Dataset/{}/{}_kegg.csv".format(args.dataset, args.dataset)
        pathway_mask=pathway2bool(path,genes)*0+1

    if torch.cuda.is_available():
        pathway_mask = pathway_mask.cuda()


    In_Nodes = pathway_mask.shape[1]
    Pathway_Nodes = pathway_mask.shape[0]

    seed=56
   
    print(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark =True


    [train_index, test_index] = Split_Sets_10_Fold(args.total_fold, np.array(data_all))

    c_index_train10, c_index_test10=[],[]
    for i in range(args.total_fold):
        print("Fold",i+1,"------------")
        x_train, ytime_train, yevent_train = torch.from_numpy(np.array(data_all)[train_index[i],:-2]).type(dtype),torch.from_numpy(np.array(data_all)[train_index[i],-1]).type(dtype).reshape(-1,1),torch.from_numpy(np.array(data_all)[train_index[i],-2]).type(dtype).reshape(-1,1)
        x_test, ytime_test, yevent_test = torch.from_numpy(np.array(data_all)[test_index[i],:-2]).type(dtype),torch.from_numpy(np.array(data_all)[test_index[i],-1]).type(dtype).reshape(-1,1),torch.from_numpy(np.array(data_all)[test_index[i],-2]).type(dtype).reshape(-1,1)


        if torch.cuda.is_available():
            x_train, ytime_train, yevent_train = x_train.cuda(), ytime_train.cuda(), yevent_train.cuda()
            x_test, ytime_test, yevent_test = x_test.cuda(), ytime_test.cuda(), yevent_test.cuda()


        #training
        net = PathExpSurv(In_Nodes, Pathway_Nodes, pathway_mask,0)
        net.sc1.weight.data = torch.rand_like(net.sc1.weight.data)#Random Initialization
        print(net)

        #if gpu is being used
        if torch.cuda.is_available():
            net.cuda()
        opt = optim.Adam(net.parameters(), lr=args.lr)
        


        main_loss_all=[]
        c_index_all=[]
        test_cindex_all =[]

        reg_out_all= []

        for epoch in range(args.num_epochs):
            net.train()
            opt.zero_grad()

            pred = net(x_train[:,:In_Nodes], yevent_train)
            main_loss = neg_par_log_likelihood(pred, ytime_train, yevent_train)

            main_loss_all.append(main_loss)

            reg_out=torch.sum(torch.abs(net.sc2.weight[pathway_mask==0]))
            reg_out_all.append(reg_out)
            alpha=0

            if args.model=='pathexpsurv':
                if epoch<100:
                    loss = main_loss + args.lambda_*torch.std(net.sc1.weight[pathway_mask>0])
                else:
                    loss = main_loss + 0.001*args.mu*reg_out
                if epoch==100:
                    net.beta = 1
                    net.sc2.weight.data=net.sc2.weight.data*0
            elif args.model=='full':
                loss = main_loss
            else:
                loss = main_loss + args.lambda_*torch.std(net.sc1.weight[pathway_mask>0]) 

            
            
            if epoch==100:
                for param_group in opt.param_groups:
                    param_group["lr"]=0.0001

            if 100<epoch<=120:
                for param_group in opt.param_groups:
                    param_group["lr"]=0.002*int((epoch-100)/4)
            if epoch>120:
                for param_group in opt.param_groups:
                    param_group["lr"]=0.01*np.power(0.95,epoch-120)



            train_cindex = c_index(pred, ytime_train, yevent_train)
            c_index_all.append(train_cindex)

            loss.backward()
            opt.step()
            with torch.no_grad():
                net.eval()
                test_cindex = c_index(net(x_test[:,:In_Nodes], yevent_test), ytime_test, yevent_test)
                test_cindex_all.append(test_cindex)

            #Positive constraint
            net.sc1.weight.data.clamp_(0)
            net.sc2.weight.data.clamp_(0)
            #Pathways mask
            net.sc1.weight.data = net.sc1.weight.data.mul(net.pathway_mask)
            net.sc2.weight.data = net.sc2.weight.data.mul(-net.pathway_mask+1)


            print("Epoch", epoch, "  main({})-reg({})".format(round(main_loss.detach().cpu().item(), 5),round(reg_out.detach().cpu().item(), 5)),
                  "cindex:{}(train),{}(test)".format(round(train_cindex.detach().item(), 5),
                                                     round(test_cindex.detach().item(), 5)),
                  "lr:{}".format(opt.param_groups[0]["lr"]))
        print("Fold",i+1,max(c_index_all),max(test_cindex_all))
        c_index_train10.append(max(c_index_all).item())
        c_index_test10.append(max(test_cindex_all).item())


        #Save pathways mask
        if not os.path.exists("Results/10cv/{}".format(args.model)):
            os.makedirs("Results/10cv/{}".format(args.model))

        pd.DataFrame(net.sc1.weight.data+net.sc2.weight.data, columns=genes).to_csv("Results/10cv/{}/".format(args.model)+ "{}_{}_pathway_bio2.csv".format(i+1,args.dataset),index=0)

    print(c_index_train10)
    print(c_index_test10)
    result=np.stack([c_index_train10,c_index_test10])
    pd.DataFrame(result).to_csv("Results/10cv/{}/".format(args.model)+args.dataset+"_10cv_{}.csv".format(args.model))





