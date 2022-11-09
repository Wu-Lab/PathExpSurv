import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import argparse
from utils import *
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.offsetbox import AnchoredText



parser = argparse.ArgumentParser(description='Downstream Analysis')
parser.add_argument('--dataset',type=str,default='BRCA',help='available datasets: BRCA | THCA | LGG')
parser.add_argument('--alpha',type=float,default=0.2,help='pathway expansion rate')
args=parser.parse_args()

data_all=pd.read_csv("Dataset/{}/{}_pro_select.csv".format(args.dataset,args.dataset))
genes = np.array(data_all.columns[:-2])
dtype = torch.FloatTensor

def data_processed(path):
    data = pd.read_csv(path)
    data_bool = np.array(data.notnull())
    data_pro = [list(np.array(data)[i][data_bool[i]][1:]) for i in range(data_bool.shape[0])]
    return data_pro


def pathway2bool(data_pro):
    W_bool=[]
    for i in range(len(data_pro)):
        W_bool.append(list(np.sum([ list(genes==x) for x in data_pro[i]],axis=0)))
    return np.array(W_bool)


def pathway_expansion(dataset,alpha):
    Begin=56
    End=156

    for seed in range(Begin,End):
        data1 = data_processed("Results/{}/Train_Valid/seed_{}/pathway_bio1.csv".format(dataset,seed))
        data2=data_processed("Results/{}/Train_Valid/seed_{}/pathway_bio2.csv".format(dataset,seed))
        if seed==Begin:
            W_stat1=pathway2bool(data1)
            W_stat2 = pathway2bool(data2)
        else:
            W_stat1+=pathway2bool(data1)
            W_stat2 += pathway2bool(data2)
    pathway_kegg = np.array(pd.read_csv("Dataset/{}/{}_kegg.csv".format(dataset,dataset)))[:, 1]
    pathway_kegg_split = []
    for x in pathway_kegg:
        pathway_kegg_split.append(x.split("/")[1:])
    pathway_mask = pathway2bool(pathway_kegg_split)

    p=alpha
    p_value=np.sort((W_stat2).reshape(-1))[-int(np.sum(pathway_mask)*(p+1))]
    print("Origin Pathways:", list(np.sum(pathway_mask,axis=1)))
    print("Union Pathways({}%) :".format(int(100+p * 100)),list(np.sum(W_stat2 >= p_value, axis=1)))

    Bool2=(W_stat2 >= p_value)

    pathway100=[]

    for i in range(Bool2.shape[0]):
        pathway100.append(list(set(genes[Bool2[i]])))

    pd.DataFrame(pathway100).to_csv("Results/{}/adjusted_pathway.csv".format(args.dataset))




def ks_test():

    dtype = torch.FloatTensor
    data_all=pd.read_csv("Dataset/LGG/LGG_pro_select.csv")
    genes = np.array(data_all.columns[:-2])
    path="Dataset/LGG/LGG_kegg.csv"
    pathway_split=pathway_kegg_split(path)
    in_prior=set([y for x in pathway_split for y in x])
    out_prior=set(genes)-in_prior

    Begin=56
    End=156
    recover_count=[]
    out_prior_count=[]

    for path in range(5):
        print(path)
        for x in pathway_split[path]:
            print(x)
            for seed in range(Begin,End):
                data = data_processed("Results/LGG/one_out/path{}/{}/seed_{}/pathway_bio2.csv".format(path,x,seed))
                if seed==Begin:
                    W_stat = pathway2bool(data)
                else:
                    W_stat+=pathway2bool(data)
            recover_count.append(W_stat[path,list(genes).index(x)]/100)
            out_prior_count.append(W_stat[path,[list(genes).index(y) for y in out_prior]]/100)

    pd.DataFrame(recover_count).to_csv("Results/LGG/one_out/recover_count.csv")
    pd.DataFrame(out_prior_count).to_csv("Results/LGG/one_out/out_prior_count.csv")


    a=np.array(pd.read_csv("Results/LGG/one_out/recover_count.csv"))[:,1]
    b=np.array(pd.read_csv("Results/LGG/one_out/out_prior_count.csv"))[:,1:]
    from scipy.stats import ks_2samp
    ks_2samp(a,b.reshape(-1))

    n_bins = 150

    # plot the cumulative histogram
    plt.figure(1,figsize=(10*0.7,6.5*0.7))
    n, bins, patches = plt.hist(a, n_bins, density=True, histtype='step',
                            cumulative=False, label='Leave-One-Out Genes')
    n, bins, patches = plt.hist(b.reshape(-1), n_bins, density=True, histtype='step',
                            cumulative=False, label='Non-Prior Genes')
    plt.text(x=0.7,y=11,s="p = 0.00004582")

    plt.legend()
    plt.title("Recoveribility Testing Histogram")
    plt.xlabel("Estimated Recovering Probability")
    plt.ylabel("Likelihood of Occurrence")
    plt.savefig("Results/LGG/one_out/ks_test.svg")



    ##rank

    rank_one_out=[]
    for k in range(69):
        rank=1065-np.where(np.sort(np.concatenate([np.array([a[k]]),b[k]]))==a[k])[0][-1]
        rank_one_out.append(rank)
        
    uniform=np.array(list(range(1065)))+1


    from scipy import stats
    stats.ks_2samp(rank_one_out,uniform)

    n_bins = 213
    plt.figure(3,figsize=(10*0.7,6.5*0.7))
    n, bins, patches = plt.hist(rank_one_out, n_bins, density=True, histtype='step',
                            cumulative=True, label='Leave-One-Out Genes')
    n, bins, patches = plt.hist(uniform, n_bins, density=True, histtype='step',
                            cumulative=True, label='Non-Prior Genes')
    plt.text(x=200,y=0.8,s="p = 0.00001038")

    plt.legend(loc='upper left')
    plt.title("Recoveribility Testing Cumulative Distribution")
    plt.xlabel("Recovering Rank")
    plt.ylabel("Likelihood of Occurrence")
    plt.savefig("Results/LGG/one_out/ks_test_rank_cumsum.svg")


def analysis_two_phase(path,start=56,end=156,mode='min'):
    loss = []
    cindex_train = []
    cindex_test = []
    for i in range(start, end):
        data = pd.read_csv(path+"Train_Valid/seed_{}/Result.csv".format(i))
        if mode=='last':
            loss.append([np.array(data)[0, 1:][99], np.array(data)[0, 1:][199]])
            cindex_train.append([np.array(data)[1, 1:][99], np.array(data)[1, 1:][199]])
            cindex_test.append([np.array(data)[2, 1:][99], np.array(data)[2, 1:][199]])
        elif mode=='min':
            loss.append([min(np.array(data)[0, 1:][:100]), min(np.array(data)[0, 1:][100:])])
            cindex_train.append([np.max(np.array(data)[1, 1:][:100]), np.max(np.array(data)[1, 1:][100:])])
            cindex_test.append([np.max(np.array(data)[2, 1:][:100]), np.max(np.array(data)[2, 1:][100:])])
        elif mode=='last50':
            loss.append([np.mean(np.array(data)[0, 1:][70:100]), np.mean(np.array(data)[0, 1:][170:])])
            cindex_train.append([np.mean(np.array(data)[1, 1:][70:100]), np.mean(np.array(data)[1, 1:][170:])])
            cindex_test.append([np.mean(np.array(data)[2, 1:][70:100]), np.mean(np.array(data)[2, 1:][170:])])
    print("Loss(100):{}±{} Loss(200):{}±{}".format('%.5f'%np.mean(loss, axis=0)[0], '%.5f'%np.std(loss, axis=0)[0],'%.5f'%np.mean(loss, axis=0)[1], '%.5f'%np.std(loss, axis=0)[1]))
    print("cindex_train(100):{}±{} cindex_train(200):{}±{}".format('%.5f'%np.mean(cindex_train, axis=0)[0], '%.5f'%np.std(cindex_train, axis=0)[0],'%.5f'%np.mean(cindex_train, axis=0)[1], '%.5f'%np.std(cindex_train, axis=0)[1]))
    print("cindex_test(100):{}±{} cindex_test(200):{}±{}".format('%.5f'%np.mean(np.array(cindex_test)[(np.array(cindex_test)>0)[:,0],:], axis=0)[0], '%.5f'%np.std(np.array(cindex_test)[(np.array(cindex_test)>0)[:,0],:], axis=0)[0],'%.5f'%np.mean(np.array(cindex_test)[(np.array(cindex_test)>0)[:,0],:], axis=0)[1], '%.5f'%np.std(np.array(cindex_test)[(np.array(cindex_test)>0)[:,0],:], axis=0)[1]))

def plot_two_phase(path,start=56,end=156):
    plt.figure(1)
    for i in range(start, end):
        data = pd.read_csv(path+"Train_Valid/seed_{}/Result.csv".format(i))
        plt.plot(np.array(data)[0, 1:], color='b', alpha=0.05, linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LOSS ({})".format(path.split('/')[1]))
    plt.savefig(path+"train_loss({}).svg".format(path.split('/')[1]))
    plt.show()

    plt.figure(2)
    for i in range(start, end):
        data = pd.read_csv(path+"Train_Valid/seed_{}/Result.csv".format(i))
        if i == start:
            plt.plot(np.array(data)[1, 1:2], color='b', label="Train", alpha=1, linestyle='--')
            plt.plot(np.array(data)[2, 1:2], color='r', label="Test", alpha=1, linestyle=':')
        else:
            plt.plot(np.array(data)[1, 1:], color='b', alpha=0.05, linestyle='--')
            plt.plot(np.array(data)[2, 1:], color='r', alpha=0.05, linestyle=':')

    plt.xlabel("Epoch")
    plt.ylabel("C-index")
    plt.title("C-INDEX ({})".format(path.split('/')[1]))
    plt.legend()
    plt.savefig(path+"train_test_cindex({}).svg".format(path.split('/')[1]))
    plt.show()


##Train_Test
def plot_retrain(path,start=56,end=156,mode='min'):
    for k in range(3):
        loss_before_at_100 = []
        loss_after_at_100 = []

        c_index_before_at_100 = []
        c_index_after_at_100 = []

        c_index_before_at_5 = []
        c_index_after_at_5 = []

        plt.figure(k + 3)
        for i in range(start, end):
            Before = pd.read_csv(path+"Train_Valid/seed_{}/Result.csv".format(i))
            After = pd.read_csv(path+"Retrain/seed_{}/Result.csv".format(i))
            if mode=='last':
                if k == 0:
                    loss_before_at_100.append(np.array(Before)[k, 100])
                    loss_after_at_100.append(np.array(After)[k, -1])
                else:
                    c_index_before_at_100.append(np.array(Before)[k, 100])
                    c_index_after_at_100.append(np.array(After)[k, -1])

                    c_index_before_at_5.append(np.array(Before)[k, 5])
                    c_index_after_at_5.append(np.array(After)[k, 5])
            elif mode=='min':
                if k == 0:
                    loss_before_at_100.append(min(np.array(Before)[k, 1:101]))
                    loss_after_at_100.append(min(np.array(After)[k, 1:]))
                else:
                    c_index_before_at_100.append(max(np.array(Before)[k, 1:101]))
                    c_index_after_at_100.append(max(np.array(After)[k, 1:]))

                    c_index_before_at_5.append(max(np.array(Before)[k, 1:6]))
                    c_index_after_at_5.append(max(np.array(After)[k, 1:6]))
            elif mode=='last50':
                if k == 0:
                    loss_before_at_100.append(np.mean(np.array(Before)[k, 51:101]))
                    loss_after_at_100.append(np.mean(np.array(After)[k, 51:]))
                else:
                    c_index_before_at_100.append(np.mean(np.array(Before)[k, 51:101]))
                    c_index_after_at_100.append(np.mean(np.array(After)[k, 51:]))

                    c_index_before_at_5.append(max(np.array(Before)[k, 1:6]))
                    c_index_after_at_5.append(max(np.array(After)[k, 1:6]))

            if i == start:
                #为了让颜色不突兀，第一个用1:2
                plt.plot(np.array(Before)[k, 1:2], color='b', label="Original", alpha=1, linestyle='--')
                plt.plot(np.array(After)[k, 1:2], color='r', label="Expanded", alpha=1, linestyle=':')
            else:
                plt.plot(np.array(Before)[k, 1:101], color='b', alpha=0.05, linestyle='--')
                plt.plot(np.array(After)[k, 1:], color='r', alpha=0.05, linestyle=':')
            plt.xlabel("Epoch")

        if k == 0:
            plt.ylabel("Loss")
            plt.title("LOSS ({})".format(path.split('/')[1]))
            print("Loss(before):{}±{}".format('%.5f'%np.mean(loss_before_at_100), '%.5f'%np.std(loss_before_at_100)))
            print("Loss(after):{}±{}".format('%.5f'%np.mean(loss_after_at_100), '%.5f'%np.std(loss_after_at_100)))
        elif k == 1:
            '''if path.split("/")[1]=='BRCA':
                plt.plot(np.ones(100)*0.879451,color='k',label='Cox',alpha=1, linestyle='--')
            elif path.split("/")[1]=='THCA':
                plt.plot(np.ones(100)*0.94024,color='k',label='Cox',alpha=1, linestyle='--')
            elif path.split("/")[1]=='LGG':
                plt.plot(np.ones(100)*0.90426,color='k',label='Cox',alpha=1, linestyle='--')'''

            plt.ylabel("C-index")
            plt.title("C-INDEX-Train ({})".format(path.split('/')[1]))
            print("Train_Cindex(before at 100):{}±{}".format('%.5f'%np.mean(c_index_before_at_100), '%.5f'%np.std(c_index_before_at_100)))
            print("Train_Cindex(after at 100):{}±{}".format('%.5f'%np.mean(c_index_after_at_100), '%.5f'%np.std(c_index_after_at_100)))

            print("Train_Cindex(before at 5):{}±{}".format('%.5f'%np.mean(c_index_before_at_5), '%.5f'%np.std(c_index_before_at_5)))
            print("Train_Cindex(after at 5):{}±{}".format('%.5f'%np.mean(c_index_after_at_5), '%.5f'%np.std(c_index_after_at_5)))
        else:
            '''if path.split("/")[1]=='BRCA':
                plt.plot(np.ones(100)*0.79683,color='k',label='Cox',alpha=1, linestyle='--')
            elif path.split("/")[1]=='THCA':
                plt.plot(np.ones(100)*0.71918,color='k',label='Cox',alpha=1, linestyle='--')
            elif path.split("/")[1]=='LGG':
                plt.plot(np.ones(100)*0.88503,color='k',label='Cox',alpha=1, linestyle='--')'''
            plt.ylabel("C-index")
            plt.title("C-INDEX-Test ({})".format(path.split('/')[1]))
            print("Test_Cindex(before at 100):{}±{}".format('%.5f'%np.mean(np.array(c_index_before_at_100)[np.array(c_index_before_at_100)>0]), '%.5f'%np.std(np.array(c_index_before_at_100)[np.array(c_index_before_at_100)>0])))
            print("Test_Cindex(after at 100):{}±{}".format('%.5f'%np.mean(np.array(c_index_after_at_100)[np.array(c_index_after_at_100)>0]), '%.5f'%np.std(np.array(c_index_after_at_100)[np.array(c_index_after_at_100)>0])))

            print("Test_Cindex(before at 5):{}±{}".format('%.5f'%np.mean(np.array(c_index_before_at_5)[np.array(c_index_before_at_5)>0]), '%.5f'%np.std(np.array(c_index_before_at_5)[np.array(c_index_before_at_5)>0])))
            print("Test_Cindex(after at 5):{}±{}".format('%.5f'%np.mean(np.array(c_index_after_at_5)[np.array(c_index_after_at_5)>0]), '%.5f'%np.std(np.array(c_index_after_at_5)[np.array(c_index_after_at_5)>0])))

        plt.legend()
        if k==0:
            plt.savefig(path+"retrain_loss({}).svg".format(path.split('/')[1]))
        elif k==1:
            plt.savefig(path + "retrain_train_cindex({}).svg".format(path.split('/')[1]))
        else:
            plt.savefig(path + "retrain_test_cindex({}).svg".format(path.split('/')[1]))
        plt.show()


def get_gene_set(task):

    path1="Dataset/{}/{}_kegg.csv".format(task, task)
    pathway1 = np.array(pd.read_csv(path1))[:, 1]
    pathway11 = []
    for x in pathway1:
        pathway11.append(x.split("/")[1:])


    path2="Results_pmt/{}/adjusted_pathway.csv".format(task)
    pathway2 = pd.read_csv(path2)
    pathway22= np.array(pathway2.notnull())
    pathway222 = [list(np.array(pathway2)[i][pathway22[i]][1:]) for i in range(pathway22.shape[0])]


    extra=[]
    for i in range(len(pathway11)):
        extra+=list(set(pathway222[i]).difference(set(pathway11[i])))
    gene_set=set(extra)

    
    return gene_set



def single_gene_survival_curve(gene):
    data_all=pd.read_csv("Dataset/{}/{}_pro_select.csv".format(task,task))
    df=data_all[[gene,'OS','OS.time']]
    
    kmf = KaplanMeierFitter()
    p=np.sort(df[gene])[df[gene].shape[0]//2]

    ix = (df[gene]>p)

    kmf.fit(df['OS.time'][ix], df['OS'][ix], label='High-Expression')
    ax = kmf.plot()
    kmf.fit(df['OS.time'][~ix], df['OS'][~ix], label='Low-Expression')
    ax = kmf.plot(ax=ax)

    # test difference between curves   
    p_v = logrank_test(df['OS.time'][ix], df['OS.time'][~ix], event_observed_A=df['OS'][ix], event_observed_B=df['OS'][~ix]).p_value
    # add p-value to plot
    ax.add_artist(AnchoredText("p = %.8f"%p_v, loc=3, frameon=False))
    plt.title(gene)
    plt.savefig("Results/Downstream_Analysis/surivial_curve_figure/{}/{}.svg".format(task,gene))



if __name__=='__main__':

    #pathway_expansion
    pathway_expansion(args.dataset,args.alpha)

    #Kolmogorov-Smirnov test
    ks_test()

    #Two phase result analysis
    path = "Results_pmt/{}/".format(args.dataset)
    analysis_two_phase(path,start=56,end=156,mode='min')
    plot_two_phase(path,start=56,end=156)
    plot_retrain(path,start=56,end=156,mode='min')

    #Single gene survival analysis
    gene_set=get_gene_set(args.dataset)
    k=1
    for gene in gene_set:
        plt.figure(k)
        single_gene_survival_curve(gene)
        k=k+1












