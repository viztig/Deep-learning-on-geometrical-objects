import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
import os
import sys
import open3d as o3d
import pandas as pd
import sklearn
from sklearn.neighbors import KDTree
import torchgeometry as tgm
import warnings
warnings.filterwarnings('ignore')#default,ignore

device=torch.device('cpu')
if torch.cuda.is_available():device=torch.device('cuda')

def normalize_pc(pcd):
    '''
    This function normalizes the point cloud by transforming
    the coordinate space to fit inside a sphere with unit radius
    '''
    points=torch.tensor(np.asarray([pcd.points]),dtype=torch.float32)
    centroid = torch.mean(points, axis=1)
    points -= centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance
    points=points.reshape(1,points.shape[2],points.shape[1])
    return points,centroid,furthest_distance
def denormalize_pc(_points,centroid,furthest_distance):
    '''
    This function denormalizes the normalized point cloud to actual coordinate space
    Multiplication with 1000 is done so that the output coordinates
    do not suffer underflow
    '''
    points=torch.clone(_points)
    points[0]*=furthest_distance
    points[1]*=furthest_distance
    points[2]*=1000
    points[0]+=centroid[0][0]
    points[1]+=centroid[0][2]
    return points

def kdtree(pcd):
    _pcd=pcd.squeeze().T
    np.random.seed(0)
    tree = KDTree(_pcd)
    nearest_dist, nearest_ind = tree.query(_pcd, k=10)#KNN_size(k)=10 
    return _pcd[nearest_ind]

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
    self.dropout=dropout
    self.fc=nn.Linear(in_dim,out_dim)
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
  
class EdgeConv(nn.Module):
  '''
  This class defines Edge Convolution layer
  '''
  def __init__(self,in_dim,out_dim):
    super().__init__()
    self.mlp=mlp(in_dim,out_dim)
  def forward(self,x):
    xknn=kdtree(x).to(device)#shape=(N,KNN_size,in_dim=3)
    xknn=xknn.reshape(xknn.shape[0],xknn.shape[2],xknn.shape[1])#shape=(N,outdim,KNN_size)
    xg=self.mlp(xknn)
    xpool=torch.max(xg,dim=2,keepdim=True)[0].squeeze()
    return xpool#shape=(N,outdim)      
  
class HPE(nn.Module):
    '''
    This class defines complete human pose network which outputs 3 predicted positions(in our case x,z coordinate of H-point and torso angle)
    More details can be found in HPE section of the README file
    '''
    def __init__(self,indim,outdim,classes=3):
        super().__init__()
        self.tnet3=tnet3()
        self.econv1=EdgeConv(indim,outdim).to(device)
        self.econv2=EdgeConv(outdim,outdim).to(device)
        self.mlp=mlp(1,1024).to(device)
        self.tnet=tnet3()   
        self.fc1=fc(1024,512)
        self.fc2=fc(512,256)
        self.fc3=fc(256,classes)
    def unit(self,_x):
        x=_x.clone()
        x1=self.econv1(x)#shape=(N,128)
        _x1=x1.clone()
        _x1=torch.tensor(_x1.T.unsqueeze(dim=0),dtype=torch.float32).to('cpu')
        x2=self.econv2(_x1)#shape=(N,128)
        xcat=torch.cat((x1,x2),dim=1).unsqueeze(dim=2)
        xpool=torch.max(xcat,dim=1,keepdim=True)[0]#shape=(N,1)
        return torch.tensor(self.mlp(xpool),dtype=torch.float32).squeeze()
    def forward(self,_x):
        x=_x.clone().to(device)
        xknn1=kdtree(_x).to(device)
        t3=self.tnet(x)
        xn=torch.matmul(t3,x).reshape(1,xknn1.shape[2],xknn1.shape[0])#shape=(N,dim)
        xn=torch.tensor(xn,dtype=torch.float32).to('cpu')
        #xn=torch.tensor(xn,dtype=torch.float32)
        xu1=self.unit(xn)
        xu1=xu1.reshape(1,xu1.shape[1],xu1.shape[0])
        xu1=torch.tensor(xu1,dtype=torch.float32).to('cpu')
        #xu1=torch.tensor(xu1,dtype=torch.float32)
        #xu2=self.unit(xu1)
        xu1=torch.max(xu1.squeeze(),1,keepdim=True)[0].squeeze().to(device)
        return self.fc3(self.fc2(self.fc1(xu1)))

def list_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

