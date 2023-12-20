from spec import *
model_val='P'
if model=="Pointnet_cls":
  from pointnet import *
  model_train=Pointnet_cls()
if model=="Pointnet_seg":
  from pointnet_seg import *
  from pointnet_seg import Pointnet_seg
  model_train=Pointnet_seg()
if model=="Pointnet_PP":
  from pointnet_PP import *
  model_train=FullModel()
if model=="GAPnet_cls":
  from gapnet_cls import *
  model_train=GAPnet_cls()
if model=="GAPnet_seg":
  from gapnet_seg import *
  model_train=GAPnet_seg()
if model=="HPE":
  from hpe import *
  model_train=HPE(3,128)

device = torch.device('cpu')
if torch.cuda.is_available():device = torch.device('cuda')

df=pd.read_excel(target_file_path)
df.set_index("Name", inplace = True)
file_paths = list_files(directory_path)

if model=="Pointnet_cls" or model=="Pointnet_seg" or model=="Pointnet_PP":
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
      print(f"Using device:{device}")
      print("Starting Training")
      self.model.train()
      for ep in range(self.epochs):
        total_epoch_loss=0
        for file in file_paths:
          pcd=o3d.io.read_point_cloud(file)
          mat,centroid,dist=normalize_pc(pcd)
          mat,centroid,dist=mat.to(device),centroid.to(device),dist.to(device)
          res=df.loc[file[len(directory_path)+1:-4]]
          material=torch.tensor([res['BI'],res['CI'], res['BB'],res['CB'],res['BP'],res['CP'],res['V']],dtype=torch.float32)
          material/=10
          material=material.to(device)
          trg=torch.tensor((torch.from_numpy(np.asarray([res['XS'],res['ZS'],res['TS']]))),dtype=torch.float32,requires_grad=True).to(device)
          output=denormalize_pc(self.model(mat,material),centroid,dist).to(device)
          self.optimizer.zero_grad()
          loss=self.loss_fn(output,trg)
          total_epoch_loss+=loss.item()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1)
          self.optimizer.step()
        print(f"Epoch:[{ep}/{self.epochs}] Epoch Loss:[{total_epoch_loss}]")
elif model=="GAPnet_cls":
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
          mat,centroid,dist=normalize_pc(pcd)
          xknn=kdtree(mat).to(device)
          mat,centroid,dist=mat.to(device),centroid.to(device),dist.to(device)
          res=df.loc[file[len(directory_path)+1:-4]]
          trg=torch.tensor((torch.from_numpy(np.asarray([res['XS'],res['YS'],res['ZS']]))),dtype=torch.float32,requires_grad=True).to(device)
          output=denormalize_pc(self.model(xknn,mat),centroid,dist)
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
else:
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
          mat,centroid,dist=normalize_pc(pcd)
          centroid,dist=centroid.to(device),dist.to(device)
          res=df.loc[file[len(directory_path)+1:-4]]
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
model_train=model_train.to(device)
_train=Train(model_train,1000,learning_rate=0.00001)
_train.train()