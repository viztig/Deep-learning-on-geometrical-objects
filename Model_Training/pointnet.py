import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
import os
import sys
import open3d as o3d
import pandas as pd
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')
if torch.cuda.is_available():device = torch.device('cuda')

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

def list_files(directory):
    '''
    this function reads the directory which contains the point cloud(.ply) files to be used for training
    '''
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

class Pointnet_cls(nn.Module):
  '''
  Pointnet class defines the modified Pointnet classification architecure
  Read the Pointnet section in README file to get more details
  This class outputs the x,z coordinate and torso angle
  '''
  def __init__(self,classes=3):
    super().__init__()
    self.tnet3=tnet3()
    self.tnet64=tnet64()
    self.mlp1=mlp(3,64)
    self.mlp2=mlp(64,64)
    self.mlp3=mlp(64,128)
    self.mlp4=mlp(128,1024)
    #self.mlp5=mlp(1024,512)
    #self.mlp6=mlp(512,256)
    #self.mlp7=mlp(256,512)
    self.Material=Material()
    self.fc1=fc(1536,2048)
    self.fc2=fc(2048,1024)
    self.fc3=fc(1024,classes)
  def forward(self,x,material):
    #input transform:
    x_=x.clone().to(device)
    t3=self.tnet3(x_)
    x=torch.matmul(t3,x)#output size=(batch_size,3,n)

    #mlp(64,64):
    x=self.mlp2(self.mlp1(x)) #output size=n*64

    #feature transform:
    x_=x.clone()
    t64=self.tnet64(x_)
    x=torch.matmul(t64,x)#output size=(batch_size,64,n)

    #mlp(64,128,1024):
    x=self.mlp4(self.mlp3(x))#output size=n*64

    x=torch.max(x,2,keepdim=True)[0]

    #mlp(512,256,k_classes):
    #x=self.mlp7(self.mlp6(self.mlp5(x)))
    x=x.squeeze()#shape=[1024]
    material=self.Material(material)
    xres=torch.cat((x,material),dim=0)
    return self.fc3(self.fc2(self.fc1(xres)))