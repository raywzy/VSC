# coding=utf-8

import json
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
import time
from Models.GCN import GCN
from torch.optim import lr_scheduler
from Tools.Wasserstein import SinkhornDistance
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random

def l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)

def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


#####  L2+Chamfer-Distance

def CDVSc(a,b,device,n,m,opts):


    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n-m) * 2) ## L2_Loss of seen classes

    #### Start Calculating CD Loss

    CD_loss=None

    A=a[n-m:]
    B=b[n-m:]

    A=A.cpu()
    B=B.cpu()

    for x in A:
        for y in B:
            dis=((x-y)**2).sum()


    for x in A:
        MINI=None
        for y in B:
            dis=((x-y)**2).sum()
            if MINI is None:
                MINI=dis
            else:
                MINI=min(MINI,dis)
        if CD_loss is None:
            CD_loss=MINI
        else:
            CD_loss+=MINI

    for x in B:
        MINI=None
        for y in A:
            dis=((x-y)**2).sum()
            if MINI is None:
                MINI=dis
            else:
                MINI=min(MINI,dis)
        if CD_loss is None:
            CD_loss=MINI
        else:
            CD_loss+=MINI


    CD_loss=CD_loss.to(device)
    #######

    lamda=0.0003

    tot_loss=L2_loss+CD_loss*opts.lamda
    return tot_loss

#####

def BMVSc(a,b,device,n,m,opts):

    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n-m) * 2) ## L2_Loss of seen classes


    A=a[n-m:]
    B=b[n-m:]

    DIS=torch.zeros((m,m))


    DIS=DIS.to(device)

    for A_id,x in enumerate(A):
        for B_id,y in enumerate(B):
            dis=((x-y)**2).sum()
            DIS[A_id,B_id]=dis

    matching_loss=0

    cost=DIS.cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    for i,x in enumerate(row_ind):
        matching_loss+=DIS[row_ind[i],col_ind[i]]

    tot_loss=L2_loss+matching_loss*opts.lamda

    return tot_loss


def WDVSc(a,b,device,n,m,opts):


    WD=SinkhornDistance(0.01,1000,None,"mean")

    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n-m) * 2) ## L2_Loss of seen classes

    A = a[n - m:]
    B = b[n - m:]

    A=A.cpu()
    B=B.cpu()
    if opts.no_use_VSC:
        WD_loss=0.
        P=None
        C=None
    else:
        WD_loss,P,C=WD(A,B)
        WD_loss = WD_loss.to(device)


    tot_loss=L2_loss+WD_loss*opts.lamda

    return tot_loss,P,C




def get_train_center(url):

    obj=json.load(open(url,"r"))
    VC=obj["train"]
    return VC


def get_cluster_center(url):

    obj=json.load(open(url,"r"))
    test_center=obj["VC"]
    return test_center




def get_attributes(device,att_url,class_url,train_class,test_class):

    attributes=[]
    with open(att_url,"r") as f:
        for lines in f:
            line=lines.strip().split()
            cur=[]
            for x in line:
                y=float(x)
                y=y/100.0
                if y<0.0:
                    y=0.0
                cur.append(y)
            attributes.append(cur)
    ys={}
    pos=0
    with open(class_url,"r") as f:
        for lines in f:
            line=lines.strip().split()
            ys[line[1]]=attributes[pos]
            pos+=1


    ret=[]
    with open(train_class,"r") as f:
        for lines in f:
            line = lines.strip().split()
            ret.append(ys[line[0]])

    with open(test_class,"r") as f:
        for lines in f:
            line = lines.strip().split()
            ret.append(ys[line[0]])


    ret=torch.tensor(ret)
    ret=ret.to(device)

    return ret




if __name__=='__main__':

    ### Fix the random seed
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)


    parser=argparse.ArgumentParser(description="Training parameters setting")
    parser.add_argument('--data_path',type=str,default="/home/ziyu/zsl_data")
    parser.add_argument('--method',type=str,default='VCL',help="VCL|CDVSc|BMVSc|WDVSc")
    parser.add_argument('--GPU',type=str,default="0")
    parser.add_argument('--dataset',type=str,default="AwA1|AwA2|CUB|SUN|SUN10")
    parser.add_argument('--split_mode',type=str,default="standard_split")
    parser.add_argument('--save_dir',type=str,default="where to save the generated target center")
    parser.add_argument('--lamda',type=float,default=0.001)
    parser.add_argument('--hidden_layers',type=str,default="2048,2048",help="define the projection network")
    parser.add_argument('--train_center',type=str,default='',help="json file which saves the VC of seen class")
    parser.add_argument('--cluster_center',type=str,default='',help='json file which saves the cluster VC of unseen class')
    parser.add_argument('--no_use_VSC',action='store_true')
    args=parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.method=='VCL':
        args.no_use_VSC=True

    yy = vars(args)
    for k, v in sorted(yy.items()):
        print('%s: %s' % (str(k), str(v)))


    device=torch.device("cuda:"+args.GPU)

    if args.dataset=="CUB":
        input_dim=312
        n=200
        m=50
        attributes_url = os.path.join(args.data_path,"CUB_200_2011/attributes/class_attribute_labels_continuous.txt")
        all_class_url = os.path.join(args.data_path, "CUB_200_2011/classes.txt")
        if args.split_mode=="standard_split":
            train_class_url=os.path.join(args.data_path,"standard_split/CUB/trainvalclasses.txt")
            test_class_url = os.path.join(args.data_path, "standard_split/CUB/testclasses.txt")
        else:
            train_class_url=os.path.join(args.data_path,"proposed_split/CUB/trainvalclasses.txt")
            test_class_url = os.path.join(args.data_path, "proposed_split/CUB/testclasses.txt")


    if args.dataset=="AwA2":
        input_dim=85
        n=50
        m=10
        attributes_url = os.path.join(args.data_path,"Animals_with_Attributes2/predicate-matrix-continuous.txt")
        all_class_url = os.path.join(args.data_path, "Animals_with_Attributes2/classes.txt")
        if args.split_mode=="standard_split":
            train_class_url=os.path.join(args.data_path,"standard_split/AWA2/trainvalclasses.txt")
            test_class_url = os.path.join(args.data_path, "standard_split/AWA2/testclasses.txt")
        else:
            train_class_url=os.path.join(args.data_path,"proposed_split/AWA2/trainvalclasses.txt")
            test_class_url = os.path.join(args.data_path, "proposed_split/AWA2/testclasses.txt")

    if args.dataset=="AwA1":
        input_dim=85
        n=50
        m=10
        attributes_url = os.path.join(args.data_path,"Animals_with_Attributes2/predicate-matrix-continuous.txt")
        all_class_url = os.path.join(args.data_path, "Animals_with_Attributes2/classes.txt")
        if args.split_mode=="standard_split":
            train_class_url=os.path.join(args.data_path,"standard_split/AWA1/trainvalclasses.txt")
            test_class_url = os.path.join(args.data_path, "standard_split/AWA1/testclasses.txt")
        else:
            train_class_url=os.path.join(args.data_path,"proposed_split/AWA1/trainvalclasses.txt")
            test_class_url = os.path.join(args.data_path, "proposed_split/AWA1/testclasses.txt")

    if args.dataset=="SUN10":
        input_dim=102
        n=717
        m=10
        attributes_url = os.path.join(args.data_path,"SUN/semantic.txt")
        all_class_url = os.path.join(args.data_path, "SUN/class.txt")
        train_class_url = "/home/ziyu/zsl_data/SUN/SUN10_train.txt"
        test_class_url = "/home/ziyu/zsl_data/SUN/SUN10_test.txt"

    if args.dataset=="SUN":

        input_dim=102
        n=717
        m=72
        attributes_url = os.path.join(args.data_path,"SUN/semantic.txt")
        all_class_url = os.path.join(args.data_path, "SUN/class.txt")
        if args.split_mode=="standard_split":
            train_class_url=os.path.join(args.data_path,"standard_split/SUN/trainvalclasses.txt")
            test_class_url = os.path.join(args.data_path, "standard_split/SUN/testclasses.txt")
        else:
            train_class_url=os.path.join(args.data_path,"proposed_split/SUN/trainvalclasses.txt")
            test_class_url = os.path.join(args.data_path, "proposed_split/SUN/testclasses.txt")




    att=get_attributes(device,attributes_url,all_class_url,train_class_url,test_class_url)
    word_vectors=att
    word_vectors = F.normalize(word_vectors) ## Normalize

    VC=get_train_center(args.train_center)  ## Firstly, to get the necessary training class center
    C_VC=get_cluster_center(args.cluster_center) ## Obtain the approximated VC of unseen class
    for x in C_VC:
        VC.append(x)

    VC=torch.tensor(VC)
    VC=VC.to(device)
    VC=F.normalize(VC)

    edges=[]
    edges = edges + [(u, u) for u in range(n)] ## Set the diagonal to 1

    output_dim=2048

    hidden_layers=args.hidden_layers
    Net = GCN(n, edges, input_dim, output_dim, hidden_layers,device).to(device)

    print('word vectors:', word_vectors.shape)
    print('VC vectors:', VC.shape)


    #####Parameters
    lr=0.0001
    wd=0.0005
    max_epoch=6000
    ####



    optimizer = torch.optim.Adam(Net.parameters(), lr=lr, weight_decay=wd)
    step_optim_scheduler=lr_scheduler.StepLR(optimizer,step_size=4000,gamma=0.1)

    #pos=0
    for epoch in range(max_epoch + 1):

        s=time.time()

        Net.train()
        step_optim_scheduler.step(epoch)

        syn_vc = Net(word_vectors)

        if args.method=='VCL':
            loss,_,_=WDVSc(syn_vc,VC,device,n,m,args)  ## Here we have set [--no_use_VSC] to True
        if args.method=='CDVSc':
            loss=CDVSc(syn_vc,VC,device,n,m,args)
        if args.method=='BMVSc':
            loss=BMVSc(syn_vc, VC, device,n,m,args)
        if args.method=='WDVSc':
            loss,_,_=WDVSc(syn_vc,VC,device,n,m,args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        e=time.time()
        print("Epoch %d Loss is %.5f Cost Time %.3f mins"%(epoch,loss.item(),(e-s)/60))
    #### Training

    Net.eval()
    output_vectors = Net(word_vectors)
    output_vectors = output_vectors.detach()

    file = "Pred_Center.txt"

    #pos+=1
    cur=os.getcwd()
    file=os.path.join(args.save_dir,file)
    with open(file,"w") as f:
        for i in range(m):
            x=i+n-m
            tmp=output_vectors[x].cpu()
            tmp=tmp.numpy()
            ret=""
            for y in tmp:
                ret+=str(y)
                ret+=" "
            f.write(ret)
            f.write('\n')
    #### Saving