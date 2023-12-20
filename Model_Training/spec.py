directory_path="seat_ply_cropped_10000"
target_file_path="target_points.xlsx"
model_path="saved_model_path.pth"
number_of_epochs="10000"


retrain_model_path=""

model="Pointnet_cls"
number_of_epochs="10000"
learning_rate="0.0001"
loss_function="L1"
optimizer="SGD"





'''
........DO NOT CHANGE ANYTHING FROM HERE ONWARDS........

1> directory_path:path of the directory where all the point cloud (.ply) format files are saved

2> model:model name which you want to train the model
        options available:
        1. Pointnet
        2.Pointnet_lowdim  
        3. Pointnet++
        4.Pointnet++_lowdim
        5. DGCNN
        6. GAPnet
        7. HPE
        8. Stat (not recommended)

3> model_path: path name where you want to save your model after training

4> number_of_epochs: number of itertaions to be done for training
5> learning_rate: value of the learning rate for training
                . should be below 1(always)
                . recommened to keep it low (recommended value:0.0001)
                . high values such as 0.1 will lead to oscillations in loss function and loss may not converge but start diverging (not recommended)
                . very low values such as 0.00000001 will lead to very slow convergence

6> loss_function: name of the loss function you want to use
                options:
                 1.L1
                 2.MSE
                 3.Custom_1
                 4.Custom_2
                 5.Custom_3(user defined)
                 
            Custom_1:loss=L1*absolute(predicted-target)/15
            Custom_2:loss=L1+absolute(predicted-target)/15
        In this way use can define your own custom loss functio Custom_3 by just going to the utility.py file and define your own loss function under the def Custom_3() function

7> optimizer: name of the optimizer you want to use while training
            options:
            1. SGD(recommended)
            2. Adam
        other optimizers can be searched on the pytorch web page under the optimizers section
        
DEFAULT:
directory_path="training_folder_path"
model="Pointnet"
model_path="saved_model_path.pth"

number_of_epochs="1000"
learning_rate="0.0001"
loss_function="L1"
optimizer="SGD"
'''