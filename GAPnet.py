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

KNN_size=10
F=16
M=4

device = torch.device('cpu')
if torch.cuda.is_available():device=torch.device('cuda')
print(device)

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

def kdtree(pcd):
    _pcd=pcd.squeeze().T
    np.random.seed(0)
    tree = KDTree(_pcd)
    nearest_dist, nearest_ind = tree.query(_pcd, k=10) 
    return _pcd[nearest_ind]

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
    return x.view(-1,3,3)
  
class SingleGAP(nn.Module):
    def __init__(self,dim,F):
        super().__init__()
        #org shape=(N,dim,KNNsize)
        self.dim=dim
        self.F=F
        self.mlpF=mlp(dim,F)#shape=(N,F,KNNsize)
        self.mlp1=mlp(F,1)#shape=(N,1,KNNsize)
        self.F=F
    def forward(self,xknn,x):
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
    def __init__(self,dim,M=4,F=16):
        super().__init__()
        self.gap=SingleGAP(dim,F).to(device)
        self.M=M
        self.F=F
    def forward(self,xknn,xn):
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
        
class GAPnet(nn.Module):
    def __init__(self,classes=1):
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
        return x_result

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
      c=0
      for file in file_paths:
        pcd=o3d.io.read_point_cloud(file)
        #nnd=NnD(np.asarray([pcd.points]))
        #v=nnd.max_vlen()
        mat,centroid,dist=normalize_pc(pcd)
        centroid,dist=centroid.to(device),dist.to(device)
        total_epoch_loss=0
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
        #if int(total_epoch_loss)<3 and ep>50:c=1
      #if c==1:break
      #if ep%10==0:self.lr=(self.lr)/4
      print(f"Epoch:[{ep}/{self.epochs}] Epoch Loss:[{total_epoch_loss}] lr:{self.lr}")

model = GAPnet().to(device)
_train=Train(model,1000,learning_rate=0.001)
_train.train()

model_path = 'your_model_path_here'
torch.save(model.state_dict(), model_path)
