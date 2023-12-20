import numpy as np
import pandas as pd
import open3d as o3d
import scipy
from scipy.stats import norm
import statistics
import os
import sys
import openpyxl
import warnings
warnings.filterwarnings('ignore')
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import torch
import torch.nn as nn

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
    points[0]*=furthest_distance
    points[1]*=furthest_distance
    points[2]*=1000
    points[0]+=centroid[0][0]
    points[1]+=centroid[0][2]
    return points


def list_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files
class mlp(nn.Module):
  '''mlp(multi layer perceptron) class defines a 1d convolution network with kernel_size=1'''
  def __init__(self,in_dim,out_dim,k_size=1):
    super().__init__()
    self.conv=nn.Conv1d(in_dim,out_dim,k_size)
  def forward(self,x):
    return self.conv(x)
class fc(nn.Module):
  '''fc class defines a full connected neural network with 1 hidden layer '''
  def __init__(self,in_dim,out_dim,k_size=1,dropout=False,dropout_p=0.7):
    super().__init__()
    self.fc=nn.Linear(in_dim,out_dim)
    #self.dropout=dropout
    #self.dp=nn.Dropout(p=dropout_p)
  def forward(self,x):
    return self.fc(x)

class tnet3(nn.Module):
  '''tnet3 class defines the T-net tranformation network to get 3x3 affine tranformation matrix'''
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
    iden = torch.eye(3, 3).repeat(1, 1, 1).to(device)#identity(iden) matrix used for stability
    x= x.view(-1,3,3)
    return x+iden

class tnet64(nn.Module):
  '''tnet64 class defines the T-net tranformation network to get 64x64 affine tranformation matrix'''    
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
    iden = torch.eye(64, 64).repeat(1, 1, 1).to(device)#identity(iden) matrix used for stability
    x=x.view(-1,64,64)
    return x+iden
#intermediate Pointnet layer
class Material(nn.Module):
    '''
    Material class is used to transform the material matrix:
    [backrest insert,cushion insert,backrest bolster,cushion bolster,backrest padding,cushion padding,ventilation]
    into higher dimensional space of 512 features
    '''
    def __init__(self):
        super().__init__()
        self.fc1=fc(7,32)
        self.fc2=fc(32,128)
        self.fc3=fc(128,512)
    def forward(self,x):
        return self.fc3(self.fc2(self.fc1(x)))
class Pointnet0(nn.Module):
  def __init__(self,dim,classes=3):
    super().__init__()
    self.tnet3=tnet3()
    self.tnet64=tnet64()
    self.mlp1=mlp(3,64)
    self.mlp2=mlp(64,1024)
    #self.mlp5=mlp(1024,512)
    #self.mlp6=mlp(512,256)
    #self.mlp7=mlp(256,512)
    self.mlp3=mlp(1024,dim)
    self.mlp4=mlp(1,classes)
  def forward(self,x):
    #input transform:
    x_=x.clone()
    t3=self.tnet3(x_)
    x=torch.matmul(t3,x)#output size=(batch_size,3,n)

    #mlp(64,64):
    x=self.mlp1(x) #output size=n*64

    #feature transform:
    x_=x.clone()
    t64=self.tnet64(x_)
    x=torch.matmul(t64,x)#output size=(batch_size,64,n)

    #mlp(64,128,1024):
    x=self.mlp2(x)#output size=n*64

    x=torch.max(x,2,keepdim=True)[0]

    #mlp(512,256,k_classes):
    #x=self.mlp7(self.mlp6(self.mlp5(x)))
    #x=x.squeeze()#shape=[1024]
    xd=self.mlp3(x)
    xd=xd.reshape(xd.shape[1],1,1)
    return self.mlp4(xd).squeeze() 

#final Pointnet layer
class Pointnet_PP(nn.Module):
  def __init__(self,classes=3):
    super().__init__()
    self.tnet3=tnet3()
    self.tnet64=tnet64()
    self.mlp1=mlp(3,64)
    self.mlp2=mlp(64,1024)
    #self.mlp5=mlp(1024,512)
    #self.mlp6=mlp(512,256)
    #self.mlp7=mlp(256,512)
    self.Material=Material()
    self.fc1=fc(1536,1024)
    self.fc2=fc(1024,classes)
  def forward(self,x,material):
    #input transform:
    x_=x.clone()
    t3=self.tnet3(x_)
    x=torch.matmul(t3,x)#output size=(batch_size,3,n)

    #mlp(64,64):
    x=self.mlp1(x) #output size=n*64

    #feature transform:
    x_=x.clone()
    t64=self.tnet64(x_)
    x=torch.matmul(t64,x)#output size=(batch_size,64,n)

    #mlp(64,128,1024):
    x=self.mlp2(x)#output size=n*64

    x=torch.max(x,2,keepdim=True)[0]

    #mlp(512,256,k_classes):
    #x=self.mlp7(self.mlp6(self.mlp5(x)))
    x=x.squeeze()#shape=[1024]
    material=self.Material(material)
    material=0.3*material
    x=0.7*x
    xres=torch.cat((x,material),dim=0)
    return self.fc2(self.fc1(xres))
  
def feature(points, num_points):
    '''
    farthest point sampling algorithm to sample and group the points in each successive layers
    '''
    selected_indices = [torch.randint(len(points), (1,)).item()]
    points = torch.tensor(points)

    for _ in range(1, num_points):
        distances = torch.norm(points - points[selected_indices[-1]], dim=1)
        distances[selected_indices] = 0  # Set selected indices' distances to 0
        selected_indices.append(torch.argmax(distances).item())

    return points[selected_indices]
#tranforming the tensors as per the requirements
def transform(matrix):
  '''
  This functions is just for transforming the matrix to required shape
  '''
  df=(matrix).T
  df[1]=df[1]*(-1)
  data=torch.stack((df[0],df[1],df[2]),dim=1)
  data=data.reshape(1,data.shape[1],data.shape[0])
  return data

class FullModel(nn.Module):
    def __init__(self,classes=3):
        super().__init__()
        self.Pointnet0_1=Pointnet0(5000).to(device)
        self.Pointnet0_2=Pointnet0(500).to(device)
        self.Pointnet=Pointnet_PP().to(device)
        self.Material=Material()
        self.fc1=fc(1536,1024)
        self.fc2=fc(1024,classes)
    def forward(self,x,material):
        x_=x.clone()
        xp1=self.Pointnet0_1(x_)
        xf1=feature(xp1.unsqueeze(dim=0)[0],1000)
        xf1=(xf1.unsqueeze(dim=0))
        xp2=self.Pointnet0_2(transform(xf1))
        xf2=feature(xp2.unsqueeze(dim=0)[0],100).unsqueeze(dim=0)
        xf3=transform(xf2)
        xp3=self.Pointnet(xf3,material)
        return xp3.squeeze()      