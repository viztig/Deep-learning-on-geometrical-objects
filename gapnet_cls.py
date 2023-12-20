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
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')
if torch.cuda.is_available():device=torch.device('cuda')
print(device)

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
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files
def kdtree(pcd):
    _pcd=pcd.squeeze().T
    np.random.seed(0)
    tree = KDTree(_pcd)
    nearest_dist, nearest_ind = tree.query(_pcd, k=10) 
    return _pcd[nearest_ind]

def knn(pcd,K=10):#KNN_size=10
    n=len(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    grouped_points = np.zeros((n, K, 3))
    for i in range(n):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], K)
        for id in idx:
            grouped_points[i][0]=pcd.points[id]
    return grouped_points
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
  
class SingleGAP(nn.Module):
    '''
    This class defines a single-head GAP layer
    '''
    def __init__(self,dim=3,F=16):
        super().__init__()
        #org shape=(N,dim,KNNsize)
        self.dim=dim
        self.F=F
        self.mlpF=mlp(dim,F)#shape=(N,F,KNNsize)
        self.mlp1=mlp(F,1)#shape=(N,1,KNNsize)
        self.F=F
    def forward(self,xknn,x,KNN_size=10):
        x1=x.reshape(xknn.shape[0],self.dim,1)#shape=(1,N,3)-->(N,3,1)
        x2=torch.tensor(xknn,dtype=torch.float32)#shape=(N,KNN_size,3)
        _x2=x2.reshape(xknn.shape[0],self.dim,KNN_size)#shape=(N,3,KNN_size)
        _x2=self.mlp1(self.mlpF(_x2))#shape=(N,1,KNN_size)
        x1=self.mlp1(self.mlpF(x1))#shape=(N,1,1)
        x1=x1.squeeze().unsqueeze(dim=1)#shape=(N,1)
        x2=x2.reshape(xknn.shape[0],self.dim,KNN_size)
        for i in range(KNN_size):
            _x2[:,:,i]=_x2[:,:,i]+x1
        xc=Fn.softmax(_x2)#shape=(N,1,KNN_size)
        x2=self.mlpF(x2)#shape=(N,F,KNN_size)
        x2=x2.reshape(xknn.shape[0],KNN_size,self.F)
        x_attn=torch.matmul(xc,x2).to(device)#shape=(N,1,F)]
        x2=x2.reshape(xknn.shape[0],KNN_size,self.F).to(device)
        return x_attn.squeeze(),x2
        #x_attn shape=(N,F)
        #x2 shape=(N,KNN_size,F)

class GAP(nn.Module):
    '''
    This class defines a single GAP layer comprised of multiple single-head GAP layers
    '''
    def __init__(self,dim=3,M=4,F=16):
        super().__init__()
        self.gap=SingleGAP(dim,F).to(device)
        self.M=M
        self.F=F
    def forward(self,xknn,xn,KNN_size=10):
        head_attn=torch.zeros(xknn.shape[0],self.F,self.M).to(device)
        head_graph=torch.zeros(xknn.shape[0],KNN_size,self.F,self.M).to(device)
        for i in range(self.M):
            x=self.gap(xknn,xn)
            head_attn[:,:,i]=head_attn[:,:,i]+x[0]
            head_graph[:,:,:,i]=head_graph[:,:,:,i]+x[1]
        head_attn=head_attn.reshape(xknn.shape[0],self.M*self.F).to(device)
        head_graph=head_graph.reshape(xknn.shape[0],KNN_size,self.F*self.M).to(device)
        return head_attn,head_graph
        #head_attn shape=(N,M*F)
        #head_graph shape=(N,KNN_size,M*F)
        
class GAPnet_cls(nn.Module):
    '''
    This class defines a complete GAPnet classification architecture 
    More details can be read from GAPnet_Classification section in README file
    '''
    def __init__(self,classes=3):
        super().__init__()
        self.gap=GAP().to(device)
        self.mlp1=mlp(67,64)
        self.mlp2=mlp(64,128)
        self.mlp3=mlp(128,128)
        self.mlp4=mlp(192,1024)
        self.tnet=tnet3()
        self.fc1=fc(1024,2048)
        self.fc2=fc(2048,1024)
        self.fc3=fc(1024,512)
        self.fc4=fc(512,256)
        self.fc5=fc(256,classes)
    def forward(self,xknn,_x):
        x=_x.clone().to(device)
        t3=self.tnet(x)
        xn=torch.matmul(t3,x).squeeze()#shape=(3,N)
        xn=xn.reshape(xknn.shape[0],3).to(device)#shape=(N,3)
        x_attn,x_graph=self.gap(xknn,xn)
        xc_attn=torch.cat((xn,x_attn),dim=1)
        xc_attn=xc_attn.unsqueeze(dim=2)
        xc_attn=self.mlp3(self.mlp2(self.mlp1(xc_attn))).squeeze()#shape=(N,128)
        xgraph_pool=torch.max(x_graph,1,keepdim=True)[0].squeeze()#shape=(N,M*F)
        xc=torch.cat((xc_attn,xgraph_pool),dim=1).unsqueeze(dim=2)#shape=(N,192)
        xc=self.mlp4(xc).squeeze()#shape=(N,1024)
        x_pool=torch.max(xc,0,keepdim=True)[0].squeeze()
        x_result=self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x_pool))))).to(device)
        return x_result.squeeze()