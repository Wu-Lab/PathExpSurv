import numpy as np
import pandas as pd
import torch

#pathway mask
def pathway2bool(path,genes):
    pathway_kegg = np.array(pd.read_csv(path))[:, 1]
    pathway_kegg_split = []
    for x in pathway_kegg:
        pathway_kegg_split.append(x.split("/")[1:])

    W_bool = []
    for i in range(len(pathway_kegg_split)):
        W_bool.append(list(np.sum([list(genes == x) for x in pathway_kegg_split[i]], axis=0)))

    pathway_mask = torch.tensor(np.array(W_bool))
    return pathway_mask

def pathway2bool_adjusted(path,genes):

    data = pd.read_csv(path)
    data_bool = np.array(data.notnull())
    data_pro = [list(np.array(data)[i][data_bool[i]][1:]) for i in range(data_bool.shape[0])]

    W_bool = []
    for i in range(len(data_pro)):
        W_bool.append(list(np.sum([list(genes == x) for x in data_pro[i]], axis=0)))
    return torch.tensor(np.array(W_bool))

def pathway2bool_ablation(path,genes):
    pathway_kegg = np.array(pd.read_csv(path))[:, 1]
    pathway_kegg_split = []
    for x in pathway_kegg:
        pathway_kegg_split.append(x.split("/")[1:])

    pathway_ablation=[]
    for i in range(len(pathway_kegg)):
        print(pathway_kegg_split[i][round(len(pathway_kegg_split[i])*0.9):])
        pathway_ablation.append(pathway_kegg_split[i][:round(len(pathway_kegg_split[i])*0.9)])

    W_bool = []
    for i in range(len(pathway_ablation)):
        W_bool.append(list(np.sum([list(genes == x) for x in pathway_ablation[i]], axis=0)))

    pathway_mask = torch.tensor(np.array(W_bool))
    return pathway_mask




def pathway_kegg_split(path):
    pathway_kegg = np.array(pd.read_csv(path))[:, 1]
    pathway_kegg_split = []
    for x in pathway_kegg:
        pathway_kegg_split.append(x.split("/")[1:])
    return pathway_kegg_split


def pathway2bool_ablation_one_out(pathway_kegg_split,genes, g,path_id):
    pathway_kegg_split[path_id].remove(g)
    W_bool = []
    for i in range(len(pathway_kegg_split)):
        W_bool.append(list(np.sum([list(genes == x) for x in pathway_kegg_split[i]], axis=0)))

    pathway_mask = torch.tensor(np.array(W_bool))
    return pathway_mask





