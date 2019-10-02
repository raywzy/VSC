### Use training set to finetune the ImageNet pretrain model. Mainly for fine-grained dataset

import torch
import torchvision.models as models
import torch.nn as nn
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import time
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.optim import lr_scheduler
import copy
import argparse



def default_loader(path):
    return Image.open(path).convert('RGB')

def get_ResNet(device):
    ResNet=models.resnet101(pretrained=True)
    x=ResNet.fc.in_features
    print(x)
    print("****")
    ResNet.fc=nn.Linear(x,150)
    ResNet=ResNet.to(device)
    return ResNet

def get_VGG(device):
    VGG19 = models.vgg19_bn(pretrained=True)
    VGG19.classifier[6]=nn.Linear(4096,150)
    VGG19=VGG19.to(device)
    return VGG19

class ImageFolder(Dataset):
    def __init__(self,imgs,transform=None,loader=default_loader):

        self.imgs=imgs
        self.transform=transform
        self.loader=loader

    def __getitem__(self, index):
        idx,fn=self.imgs[index]
        img=self.loader(fn)
        #img=img.resize((224,224))
        if self.transform is not None:
            img=self.transform(img)
        return img,idx

    def __len__(self):
        return len(self.imgs)

def get_data_loader(opts):

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    ys={}
    pos=0
    train_class=[]
    with open(opts.train_class_list,"r") as f:
        for lines in f:
            line=lines.strip().split()
            train_class.append(line[0])
            ys[line[0]]=pos
            pos+=1

    imgs = []

    for x in train_class:
        cur_class_url=os.path.join(opts.data_dir,x)

        image_list=os.listdir(cur_class_url)
        for y in image_list:
            path=os.path.join(cur_class_url,y)
            imgs.append((ys[x],path))

    all_dataset={}
    all_dataset["train"]=ImageFolder(imgs,data_transform['train'])
    all_dataset["val"]=ImageFolder(imgs,data_transform['val'])

    L=len(all_dataset["train"])

    all_loader={}
    all_loader["train"] =DataLoader(dataset=all_dataset["train"], batch_size=32, shuffle=True,num_workers=8)
    all_loader["val"]=DataLoader(dataset=all_dataset["val"], batch_size=32, shuffle=False,num_workers=8)

    return all_loader,L



if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='',help='root of dataset')
    parser.add_argument('--train_class_list',type=str,default='',help='Train Class')
    parser.add_argument('--gpu_id',type=int,default=0)
    opts=parser.parse_args()


    device=torch.device(opts.gpu_id)

    ResNet=get_ResNet(device)
    #VGG=get_VGG(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(ResNet.parameters(),lr=0.001,momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    epoch_num=25

    Loader,length=get_data_loader(opts)

    #############################

    #start=time.time()

    best_model_wts=copy.deepcopy(ResNet.state_dict())  ### save intermediate model
    best_acc=0.0



    for epoch in range(epoch_num):

        print('Epoch {}/{}'.format(epoch,epoch_num-1))
        print("-"*10)


        for phase in ['train','val']:

            if phase=='train':
                exp_lr_scheduler.step()
                ResNet.train()
            else:
                ResNet.eval()


            running_loss = 0.0
            running_correct=0

            for images,labels in Loader[phase]:


                images=images.to(device)
                labels=labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):

                    outputs=ResNet(images)
                    _,preds=torch.max(outputs,1)
                    loss=criterion(outputs,labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss+=loss.item()*images.size(0)
                running_correct+=torch.sum(preds==labels.data)

            epoch_loss=running_loss/length
            epoch_acc=running_correct.double()/length

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase,epoch_loss,epoch_acc))

            if phase=='val':
                best_model_wts=copy.deepcopy(ResNet.state_dict())
                model_name="FT_model_"+str(epoch)+".pkl"
                torch.save(best_model_wts, model_name)


