This directory contains all the models that were used for training and evalutaion.

### Pointnet

This shows a pointnet network with classification network on top and segmentation network in bottom. 

##### Pointnet_cls

Pointnet classification network is implemented only till max pooling layer to get global feature matrix. After this the material matrix and the global feature matrix is multplied with a weight to give more influence to the geometry matrix and less to material matrix and then these matrices are concatenated and passed through a neural network with multiple hidden layers to finally obtain the x,z coordinate of H-point and torso angle

##### Pointnet_seg

Pointnet segmentation network is implemented only till max pooling layer to get the global feature matrix which is then concatenated with the material matrix(xres). After this xres is passed through a hidden layer(fc4) and concatenated with the (nx1024) size geometry matrix(x_b). This combined matrix(xres_cat) is passed through the shared mlp network of(512,256,128) and then through max pooling layer to get xres_pool matrix. This xres_pool matrix is then passed through 3 hidden layers(fc3,fc2,fc1) to get the final 3 predictions

### Pointnet++

![1702974886880](image/README/1702974886880.png)

Pointnet_PP class defines the pointnet++ classification architecture. Pointnet class defines the intermediary pointnet layers with mlp(multi layer perceptron) as the finaly layer and Pointnet__PP class defines the final pointnet layer along with a fully connected neural network which outputs the 3 predicted values. FullModel class defines the entire architecture with xp_1 and xp_2 being the two pointnet layers after which xf3 matrix is obtained which is the passed through a final Pointnet_PP layer with the material matrix to obtain the predicted results.

### GAPnet

##### Gapnet_cls

![1702976798591](image/README/1702976798591.png)

SingleGAP class defines the single-head gap layer shown in the above image and outputs graph features matrix(x2) and attention feature matrix(x_attn). For loop is used to add the two matrices of size (Nx1) and (Nxkx1) before passing it through the softmax function.

![1702977574586](image/README/1702977574586.png)

GAP class defines the GAPlayer shown in above image and outputs the multi-head graph and attention features matrix: head_graph, head_attn.For loop is used to concat the garph and attention features of 'M' single-head gap layers.

![1702977750002](image/README/1702977750002.png)

GAPnet class defines the GAPnet classification architecture which is implemented till getting the global features after which the global features matrix is passed through a fully connected neural network with 5 hidden layers(fc5,...,fc1) to output the predicted value.

##### Gapnet_seg

It uses the same singleGAP and GAP classes to define the single-head gap and GAP layer respectively

![1702978251551](image/README/1702978251551.png)

The GAPnet class defines the GAPnet segmentation architecture shown in above image which is implemented till obtaining the global feature matrix from max pooling which is then passed through a fully connected neural network with 5 hidden layers(fc5,...fc1) to obtain the predicted results.

### HPE

For spatial transformation, HPE uses tnet3 or 3x3 transformation matrix.

![1702978771127](image/README/1702978771127.png)

EdgeConv class defines the Edge convolution layer shown in above image. In my implementation, I have used kdtree algorithm inplace of the k-nn graph algorithm.

![1702979079278](image/README/1702979079278.png)

HPE class defines the complete network shown in the above image . Finally predicted results are obtained with M=3 for the x,z coordinate of H-point and the torso angle.
