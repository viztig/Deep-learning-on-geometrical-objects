import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import open3d as o3d
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#gpu usage if available
device = torch.device('cpu')
if torch.cuda.is_available():device=torch.device('cuda')
print(device)

#unit sphere normalisation
def normalize_pc(pcd):
    points=torch.tensor(np.asarray([pcd.points]),dtype=torch.float32)
    centroid = torch.mean(points, axis=1)
    points -= centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance
    #points*=10
    points=points.reshape(1,points.shape[2],points.shape[1])
    return points,centroid,furthest_distance
def denormalize_pc(_points,centroid,furthest_distance):
    points=torch.clone(_points)
    #points/=10
    points*=furthest_distance
    points[0]+=centroid[0][0]
    points[1]+=centroid[0][2]
    return points

#tranforming the tensors as per the requirements
def transform(matrix):
  df=(matrix).T
  df[1]=df[1]*(-1)
  data=torch.stack((df[0],df[1],df[2]),dim=1)
  data=data.reshape(1,data.shape[1],data.shape[0])
  return data

class mlp(nn.Module):
  def __init__(self,in_dim,out_dim,k_size=1):
    super().__init__()
    self.conv=nn.Conv1d(in_dim,out_dim,k_size)
  def forward(self,x):
    return self.conv(x)
class fc(nn.Module):
  def __init__(self,in_dim,out_dim):
    super().__init__()
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

#material layer
class Material(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=fc(7,128)
    def forward(self,x):
        return self.fc1(x)

#intermediate Pointnet layer
class Pointnet0(nn.Module):
  def __init__(self,dim,classes=2):
    super().__init__()
    self.tnet3=tnet3()
    self.tnet64=tnet64()
    self.mlp1=mlp(3,64)
    self.mlp2=mlp(64,1024)
    #self.mlp5=mlp(1024,512)
    #self.mlp6=mlp(512,256)
    #self.mlp7=mlp(256,512)
    self.mlp3=mlp(1024,dim)
    self.mlp4=mlp(1,3)
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
class Pointnet(nn.Module):
  def __init__(self,classes=2):
    super().__init__()
    self.tnet3=tnet3()
    self.tnet64=tnet64()
    self.mlp1=mlp(3,64)
    self.mlp2=mlp(64,1024)
    #self.mlp5=mlp(1024,512)
    #self.mlp6=mlp(512,256)
    #self.mlp7=mlp(256,512)
    self.Material=Material()
    self.fc1=fc(1152,1024)
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

#farthest point sampling algorithm to sample and group the points in each successive layers
def feature(points, num_points):
    selected_indices = [torch.randint(len(points), (1,)).item()]
    points = torch.tensor(points)  # Convert to PyTorch tensor if it's not already

    for _ in range(1, num_points):
        distances = torch.norm(points - points[selected_indices[-1]], dim=1)
        distances[selected_indices] = 0  # Set selected indices' distances to 0
        selected_indices.append(torch.argmax(distances).item())

    return points[selected_indices]

class FullModel(nn.Module):
    def __init__(self,classes=2):
        super().__init__()
        self.Pointnet0_1=Pointnet0(5000).to(device)
        self.Pointnet0_2=Pointnet0(500).to(device)
        self.Pointnet=Pointnet().to(device)
        self.Material=Material()
        self.fc1=fc(1152,1024)
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
        return xp3      

df=pd.read_excel("target_points.xlsx")
df.set_index("Name", inplace = True)
#function to read the folders and its content
def list_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

file_paths = list_files(directory_path)

class Train:
  def __init__(self,_model,epochs,batch_size=1,learning_rate=0.000001,momentum=0.8):
    self.batch=batch_size
    self.lr=learning_rate
    self.epochs=epochs
    self.momentum=momentum
    self.optimizer=torch.optim.SGD(_model.parameters(),lr=self.lr,momentum=self.momentum)
    #self.optimizer=torch.optim.Adam(_model.parameters(),lr=self.lr)
    self.loss_fn=nn.L1Loss()
    self.model=_model
  def train(self):
    print("starting")
    self.model.train()
    for ep in range(self.epochs):
      total_epoch_loss=0
      for file in file_paths:
        pcd=o3d.io.read_point_cloud(file)
        points,cent,dist=normalize_pc(pcd)
        mat=make_tensor_t(points)
        mat,cent,dist=mat.to(device),cent.to(device),dist.to(device)
        res=df.loc[file[20:-6]]
        material=torch.tensor([res['BI'],res['CI'], res['BB'],res['CB'],res['BP'],res['CP'],res['V']],dtype=torch.float32)
        material/=10
        material=material.to(device)
        trg=torch.tensor((torch.from_numpy(np.asarray([res['XS'],res['ZS']]))),dtype=torch.float32,requires_grad=True).to(device)
        output=denormalize_pc(self.model(mat,material),cent,dist)
        self.optimizer.zero_grad()
        loss=self.loss_fn(output,trg)
        total_epoch_loss+=loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1)
        self.optimizer.step()
      print(f"Epoch:[{ep}/{self.epochs}] Epoch Loss:[{total_epoch_loss}]")

model = FullModel().to(device)
_train=Train(model,1000,learning_rate=0.001)
_train.train()

model_path = 'your_model_path_here'
torch.save(model.state_dict(), model_path)
