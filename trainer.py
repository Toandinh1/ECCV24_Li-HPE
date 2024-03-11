
import yaml
import tensorflow as tf
import numpy as np
import torch
import os
# Please add the downloaded mmfi directory into your python project. 
from mmfi1 import make_dataset, make_dataloader, MMFi_Dataset, decode_config, MMFi_Database
import torch.nn as nn
from evaluation import compute_pck_pckh
from sklearn.model_selection import train_test_split
from evaluate import compute_similarity_transform, calulate_error
from network_twostream import posenet, weights_init
#from ConViT_network import posenet, weights_init
#from Dynam_Residual_network import posenet, weights_init
#from WiLDAR_network import posenet, weights_init
from minirocket_3variables import fit, transform
from minirocket import features
from sklearn.metrics.pairwise import cosine_similarity
from feature_multiRocket import features_multi_rocket
from features_hydra import features_hydra
from sklearn.feature_selection import VarianceThreshold
import thop 

#X = torch.rand(32,3,114,10)
metafi = posenet()
#metafi = metafi.cuda()
#flops, params = thop.profile(metafi, inputs=(X,))
#print(f"FLOPs: {flops / 1e9} GigaFLOPs")  # Chuyển đổi sang GigaFLOPs (tỷ lệ FLOPs)
#print(f"Parameters: {params / 1e6} Million")

dataset_root ='your link data'
with open('config.yaml', 'r') as fd:  # change the .yaml file in your code.
    config = yaml.load(fd, Loader=yaml.FullLoader)


train_dataset, test_dataset = make_dataset(dataset_root, config)
rng_generator = torch.manual_seed(config['init_rand_seed'])
train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
#testing_loader = make_dataloader(test_dataset, is_training=False, generator=rng_generator, **config['test_loader'])
val_data , test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)
val_loader = make_dataloader(val_data, is_training=False, generator=rng_generator, **config['val_loader'])
test_loader = make_dataloader(test_data, is_training=False, generator=rng_generator, **config['test_loader'])


metafi.apply(weights_init)
metafi = metafi.cuda()

#l2_loss = nn.L2Loss().cuda() 
criterion_L2 = nn.MSELoss().cuda()
optimizer = torch.optim.SGD(metafi.parameters(), lr = 0.001, momentum=0.9)
n_epochs = 20
n_epochs_decay = 30
epoch_count = 1
def lambda_rule(epoch):

    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    return lr_l
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1))
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1))

l2_lambda = 0.001
regularization_loss = 0
for param in metafi.parameters():
    regularization_loss += torch.norm(param, p=2)  # L2 regularization term

# Tổng loss function là tổng của hàm loss và regularization loss
def total_loss(output, target):
    loss = criterion_L2(output, target)
    reg_loss = l2_lambda * regularization_loss
    return loss + reg_loss

num_epochs = 50
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
pck_50_overall_max = 0
train_mean_loss_iter = []
valid_mean_loss_iter = []
time_iter = []


for epoch_index in range(num_epochs):

    loss = 0
    train_loss_iter = []
    metric = []
    metafi.train()
    relation_mean =[]
    for idx, data in enumerate(train_loader):

        csi_data = data['input_wifi-csi']
        
            
       
        
        
        csi_data = torch.tensor(csi_data)
        csi_data = csi_data.cuda()
        csi_data = csi_data.type(torch.cuda.FloatTensor)
        #csi_dafeaturesta = csi_data.view(16,2,3,114,10)
        keypoint = data['output']#17,3
        keypoint = keypoint.cuda()
       
        
        xy_keypoint = keypoint[:,:,0:2].cuda()
        confidence = keypoint[:,:,2:3].cuda()
        
        
        
       

        pred_xy_keypoint, time = metafi(csi_data) #b,2,17,17
        pred_xy_keypoint = pred_xy_keypoint.squeeze()
        #pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 1, 2)
        #flops, params = thop.profile(metafi, inputs=(csi_data,))
        #loss = tf.reduce_mean(tf.pow(pred_xy_keypoint - xy_keypoint, 2))
        #loss = criterion_L2(pred_xy_keypoint, xy_keypoint)/32
        loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
        #loss += l2_loss
        train_loss_iter.append(loss.cpu().detach().numpy())
        time_iter.append(time)
        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()

        lr = np.array(scheduler.get_last_lr())
        #print(f"FLOPs: {flops / 1e9} GigaFLOPs")  # Chuyển đổi sang GigaFLOPs (tỷ lệ FLOPs)
        #print(f"Parameters: {params / 1e6} Million")
        message = '(epoch: %d, iters: %d, lr: %.5f, loss: %.3f) ' % (epoch_index, idx * 32, lr, loss)
        print(message)
    scheduler.step()
    sum_time = np.mean(time_iter)
    train_mean_loss = np.mean(train_loss_iter)
    train_mean_loss_iter.append(train_mean_loss)
    #relation_mean = np.mean(relation, 0)
    print('end of the epoch: %d, with loss: %.3f' % (epoch_index, train_mean_loss,))
    #total_params = sum(p.numel() for p in metafi.parameters())
    #print("Số lượng tham số trong mô hình: ", total_params)
    #print("Tổng thời gian train: ", sum_time)
    
    metafi.eval()
    valid_loss_iter = []
    #metric = []
    pck_50_iter = []
    pck_20_iter = []
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            csi_data = data['input_wifi-csi']
            
            
            
            csi_data = torch.tensor(csi_data)
            csi_data = csi_data.cuda()
            csi_data = csi_data.type(torch.cuda.FloatTensor)
            
            #csi_dafeaturesta = csi_data.view(16,2,3,114,10)
            keypoint = data['output']#17,3
            keypoint = keypoint.cuda()
            
            
            xy_keypoint = keypoint[:,:,0:2].cuda()
            confidence = keypoint[:,:,2:3].cuda()

            pred_xy_keypoint,time = metafi(csi_data)  # 4,2,17,17
            #pred_xy_keypoint = pred_xy_keypoint.squeeze()
            #pred_xy_keypoint = pred_xy_keypoint.reshape(length_val,17,2)
            #flops, params = thop.profile(metafi, inputs=(csi_data,))
            
            #loss = tf.reduce_mean(tf.pow(pred_xy_keypoint - xy_keypoint, 2))
            loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))
            #loss = criterion_L2(pred_xy_keypoint, xy_keypoint)
            
            valid_loss_iter.append(loss.cpu().detach().numpy())
            pred_xy_keypoint = pred_xy_keypoint.cpu()
            xy_keypoint = xy_keypoint.cpu()
            #pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 0, 1).unsqueeze(dim=0)
            #xy_keypoint = torch.transpose(xy_keypoint, 0, 1).unsqueeze(dim=0)
            pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
            xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)
            #keypoint = torch.transpose(keypoint, 1, 2)
            #pred_xy_keypoint_pck = pred_xy_keypoint.cpu()
            #xy_keypoint_pck = xy_keypoint.cpu()
            pck = compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
            #mpjpe,pa_mpjpe = calulate_error(pred_xy_keypoint, xy_keypoint)
             
            metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))
            pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5))
            pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.2))
            
            
            #message1 = '( loss: %.3f) ' % (loss)
            #print(message1)
            #pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.5))
            #print(f"FLOPs: {flops / 1e9} GigaFLOPs")  # Chuyển đổi sang GigaFLOPs (tỷ lệ FLOPs)
            #print(f"Parameters: {params / 1e6} Million")
            #pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.2))

        valid_mean_loss = np.mean(valid_loss_iter)
        #train_mean_loss = np.mean(train_loss_iter)
        valid_mean_loss_iter.append(valid_mean_loss)
        mean = np.mean(metric, 0)*1000
        mpjpe_mean = mean[0]
        pa_mpjpe_mean = mean[1]
        pck_50 = np.mean(pck_50_iter,0)
        pck_20 = np.mean(pck_20_iter,0)
        pck_50_overall = pck_50[17]
        pck_20_overall = pck_20[17]
        print('validation result with loss: %.3f, pck_50: %.3f, pck_20: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f' % (valid_mean_loss, pck_50_overall,pck_20_overall, mpjpe_mean, pa_mpjpe_mean))
        
        if pck_50_overall > pck_50_overall_max:
           print('saving the model at the end of epoch %d with pck_50: %.3f' % (epoch_index, pck_50_overall))
           torch.save(metafi, 'get_train.py')
           pck_50_overall_max = pck_50_overall


        if (epoch_index+1) % 50 == 0:
            print('the train loss for the first %.1f epoch is' % (epoch_index))
            print(train_mean_loss_iter)


import matplotlib.pyplot as plt

epochs = list(range(1,num_epochs+1))
training_loss = train_mean_loss_iter
validation_loss = valid_mean_loss_iter

plt.plot(epochs, training_loss, label='Training Loss', color='blue')

# Vẽ đồ thị loss function cho tập validation
plt.plot(epochs, validation_loss, label='Validation Loss', color='red')

# Tùy chỉnh đồ thị
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Đồ thị Loss Function qua Epochs')
plt.legend()
# Hiển thị đồ thị
plt.show()             


        


        







