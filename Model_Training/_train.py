from spec import *
from pointnet import *
model_train=Pointnet_cls()

device = torch.device('cpu')
if torch.cuda.is_available():device = torch.device('cuda')

df=pd.read_excel(target_file_path)
df.set_index("Name", inplace = True)
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

_train=Train(model_train,1000,learning_rate=0.00001)
_train.train()