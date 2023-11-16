import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
import os
import sys
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')
if torch.cuda.is_available():device = torch.device('cuda')

def normalize_pc(pcd):
    points=torch.tensor(np.asarray([pcd.points]),dtype=torch.float32)
    centroid = torch.mean(points, axis=1)
    points -= centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance
    points=points.reshape(1,points.shape[2],points.shape[1])
    return points,centroid,furthest_distance
def denormalize_pc(_points,centroid,furthest_distance):
    points=torch.clone(_points)
    points*=furthest_distance
    points+=centroid[0]
    return points

class mlp(nn.Module):
  def __init__(self,in_dim,out_dim,k_size=1):
    super().__init__()
    self.conv=nn.Conv1d(in_dim,out_dim,k_size)
  def forward(self,x):
    return self.conv(x)
class fc(nn.Module):
  def __init__(self,in_dim,out_dim,k_size=1,dropout=False,dropout_p=0.7):
    super().__init__()
    self.dropout=dropout
    self.fc=nn.Linear(in_dim,out_dim)
    #self.dp=nn.Dropout(p=dropout_p)
  def forward(self,x):
    return self.fc(x)

class tnet3(nn.Module):
  def __init__(self):
    super().__init__()
    self.mlp1=mlp(3,64)
    self.mlp2=mlp(64,128)
    self.mlp3=mlp(128,1024)
    self.fc1=fc(1024,512)
    self.fc2=fc(512,256)
    self.fc3=fc(256,3*3)
  def forward(self,x):
    x=self.mlp3(self.mlp2(self.mlp1(x)))
    x=torch.max(x,2)[0] #torch.max ouputs [max,max_indices]
    x=self.fc3(self.fc2(self.fc1(x)))
    iden = torch.eye(3, 3).repeat(1, 1, 1).to(device)
    x= x.view(-1,3,3)
    return x+iden

class tnet64(nn.Module):
  def __init__(self):
    super().__init__()
    self.mlp1=mlp(64,64)
    self.mlp2=mlp(64,128)
    self.mlp3=mlp(128,1024)
    self.fc1=fc(1024,512)
    self.fc2=fc(512,256)
    self.fc3=fc(256,64*64)
  def forward(self,x):
    x=self.mlp3(self.mlp2(self.mlp1(x)))
    x=torch.max(x,2)[0]#torch.max ouputs [max,max_indices]
    x=self.fc3(self.fc2(self.fc1(x)))
    iden = torch.eye(64, 64).repeat(1, 1, 1).to(device)
    x=x.view(-1,64,64)
    return x+iden

class Material(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=fc(7,32)
        self.fc2=fc(32,128)
        self.fc3=fc(128,512)
    def forward(self,x):
        return self.fc3(self.fc2(self.fc1(x)))

class Pointnet(nn.Module):
  def __init__(self,classes=3):
    super().__init__()
    self.tnet3=tnet3()
    self.tnet64=tnet64()
    self.mlp1=mlp(3,64)
    self.mlp2=mlp(64,64)
    self.mlp3=mlp(64,128)
    self.mlp4=mlp(128,1024)
    self.mlp5=mlp(1536,100)
    #self.mlp5=mlp(1024,512)
    #self.mlp6=mlp(512,256)
    #self.mlp7=mlp(256,512)
    self.Material=Material()
    self.fc1=fc(128,512)
    self.fc2=fc(512,256)
    self.fc3=fc(256,3)
    self.mlp_s1=mlp(1600,512)
    self.mlp_s2=mlp(512,256)
    self.mlp_s3=mlp(256,128)
  def forward(self,x,material):
    #input transform:
    org=x.clone()
  
    x_=x.clone()
    t3=self.tnet3(x_)
    x=torch.matmul(t3,x)#output size=(batch_size,3,n)

    #mlp(64,64):
    x=self.mlp2(self.mlp1(x)) #output size=n*64
    x_b=x.squeeze()
    #feature transform:
    x_=x.clone()
    t64=self.tnet64(x_)
    self.x=torch.matmul(t64,x)#output size=(batch_size,64,n)

    #mlp(64,128,1024):
    x=self.mlp4(self.mlp3(x))#output size=n*64

    x=torch.max(x,2,keepdim=True)[0]

    #mlp(512,256,k_classes)
    x=x.squeeze()#shape=[1024]
    material=self.Material(material)
    xres=torch.cat((x,material),dim=0).unsqueeze(dim=1)
    fc4=fc(1,org.shape[2]).to(device)
    xres=fc4(xres)
    xres_cat=torch.cat((xres,x_b),dim=0)
    xres_cat=self.mlp_s3(self.mlp_s2(self.mlp_s1(xres_cat)))
    xres_pool=torch.max(xres_cat,1,keepdim=True)[0]
    xres_pool=xres_pool.reshape(1,xres_pool.shape[0])
    return self.fc3(self.fc2(self.fc1(xres_pool)))

df=pd.read_excel("target_points.xlsx")
df.set_index("Name", inplace = True)
def list_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

file_paths = list_files('your_directory_path_here')

class Train:
  def __init__(self,_model,epochs,batch_size=1,learning_rate=0.00001,momentum=0.8):
    self.batch=batch_size
    self.lr=learning_rate
    self.epochs=epochs
    self.momentum=momentum
    #self.optimizer=torch.optim.Adam(_model.parameters(),lr=self.lr)
    self.optimizer=torch.optim.SGD(_model.parameters(),lr=self.lr,momentum=self.momentum)
    self.loss_fn=nn.L1Loss()
    self.model=_model
  def train(self):
    print("starting")
    self.model.train()
    for ep in range(self.epochs):
      total_epoch_loss=0
      for file in file_paths:
        pcd=o3d.io.read_point_cloud(file)
        #nnd=NnD(np.asarray([pcd.points]))
        #v=nnd.max_vlen()
        mat,centroid,dist=normalize_pc(pcd)
        mat,centroid,dist=mat.to(device),centroid.to(device),dist.to(device)
        res=df.loc[file[23:-4]]
        trg=torch.tensor((torch.from_numpy(np.asarray([res['XS'],res['YS'],res['ZS']]))),dtype=torch.float32,requires_grad=True).to(device)
        output=denormalize_pc(self.model(mat),centroid,dist)
        #output=torch.tensor([_output[0],_output[2]],dtype=torch.float32,requires_grad=True).to(device)
        self.optimizer.zero_grad()
        loss=self.loss_fn(output,trg)
        total_epoch_loss+=loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1)
        self.optimizer.step()
      print(f"Epoch:[{ep}/{self.epochs}] Epoch Loss:[{total_epoch_loss}] lr:{self.lr}")

model = Pointnet().to(device)
_train=Train(model,1000,learning_rate=0.001)
_train.train()

model_path = 'your_model_path_here'
torch.save(model.state_dict(), model_path)
