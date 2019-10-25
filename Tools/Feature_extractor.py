### Extract the fetures of all category, then generate one feature matrix for each class (one json file)


import os
import torch
import torchvision.models as model
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace
from matplotlib import cm
import json
import argparse
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def loader(path):
    return Image.open(path).convert('RGB')

def transform(path):
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img=loader(path)
    img=img.resize((224,224))
    img=trans(img)
    return img


if __name__=='__main__':



    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='',help='root of dataset')
    parser.add_argument('--pretrain_model',type=str,default='',help='Use pretrained model to extract features')
    parser.add_argument('--mode',type=str,default='SS',help='SS|PS')
    parser.add_argument('--gpu_id',type=int,default=0, help='Single GPU')
    opts=parser.parse_args()

    #CUB_fn = os.path.join(root, "zsl_data", "CUB_200_2011", "images")
    #AwA2_fn=os.path.join(root,"zsl_data","Animals_with_Attributes2","JPEGImages")
    #SUN_fn=os.path.join(root,"zsl_data","SUN","image")
    #AwA1_fn="/home/ziyu/zsl_data/AwA/images"


    device=torch.device(opts.gpu_id)

    ResNet = model.resnet101(pretrained=True)
    x = ResNet.fc.in_features
    if opts.pretrain_model!="":
        ResNet.fc = nn.Linear(x, 150)
        ResNet.load_state_dict(torch.load(opts.pretrain_model)) ##"/home/ziyu/DNN_RC/Tools/CUB_FT_SS_101_22.pkl"
    ResNet = nn.Sequential(*list(ResNet.children())[:-1])
    ResNet = ResNet.to(device)
    ResNet.eval()


    data_fn=opts.data_dir

    fea = []
    label = []
    for x in os.listdir(data_fn):
        cur = os.path.join(data_fn, x)
        cur_class_list = os.listdir(cur)


        all_features=[]
        for image_id in cur_class_list:  #### Load each image of each class

            image_fn = os.path.join(cur, image_id)
            image = transform(image_fn)
            image = image.unsqueeze(0)  ### 4 dimensions input
            image = image.to(device)
            features = ResNet(image)
            f = features.to("cpu")
            f = f.detach().numpy()
            fea_vec = f[0].reshape(-1)
            # print(fea_vec.shape)
            fea_vec = torch.tensor(fea_vec)
            fea_vec = F.normalize(fea_vec, dim=0)
            fea_vec = fea_vec.numpy()
            all_features.append(fea_vec.tolist())

        obj={}
        obj["features"]=all_features
        cur_url=os.path.join(cur,"ResNet101_%s.json"%(opts.mode))
        json.dump(obj,open(cur_url,"w"))
        print("%s has finished ..."%(x))
