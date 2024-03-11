import yaml
import numpy as np
import torch
import os
from mmfi1 import make_dataset, make_dataloader, MMFi_Dataset, decode_config, MMFi_Database
import torch.nn as nn
from evaluation import compute_pck_pckh
from sklearn.model_selection import train_test_split
from evaluate import compute_similarity_transform, calulate_error
from network import posenet, weights_init
from minirocket_3variables import fit, transform
from minirocket import features
from features_HDC import HDC_feature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter
from scipy.signal import firwin, lfilter
from scipy.ndimage import uniform_filter
from scipy.signal import medfilt
# import get_data_type
dataset_root ='your link data'
with open('config.yaml', 'r') as fd:  # change the .yaml file in your code.
    config = yaml.load(fd, Loader=yaml.FullLoader)

def add_salt_and_pepper_noise(signal, noise_level):
    noisy_signal = np.copy(signal)
    num_salt = np.floor(noise_level * signal.size * 0.5)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in signal.shape]
    salt_coords = np.array(salt_coords)
    salt_coords = np.clip(salt_coords, 0, signal.shape[0]-1)
    noisy_signal[salt_coords[0], salt_coords[1], salt_coords[2], salt_coords[3]] = 1

    num_pepper = np.floor(noise_level * signal.size * 0.5)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in signal.shape]
    pepper_coords = np.array(pepper_coords)
    pepper_coords = np.clip(pepper_coords, 0, signal.shape[0]-1)
    noisy_signal[pepper_coords[0], pepper_coords[1], pepper_coords[2], pepper_coords[3]] = 0
    return noisy_signal


train_dataset, test_dataset = make_dataset(dataset_root, config)
rng_generator = torch.manual_seed(config['init_rand_seed'])
train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
#testing_loader = make_dataloader(test_dataset, is_training=False, generator=rng_generator, **config['test_loader'])
val_data , test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)
val_loader = make_dataloader(val_data, is_training=False, generator=rng_generator, **config['test_loader'])
test_loader = make_dataloader(test_data, is_training=False, generator=rng_generator, **config['test_loader'])
 
device = torch.device("cuda")
metafi = torch.load('get_train.py')
metafi = metafi.eval()

epsilon = 0.4
criterion_L2 = nn.MSELoss()
loss = 0
test_loss_iter = []
metric = []
time_iter = []
pck_50_iter = []
pck_40_iter = []
pck_30_iter = []
pck_20_iter = []
pck_10_iter = []
pck_5_iter = []
with torch.no_grad():
    for i, data in enumerate(test_loader):

        csi_data = data['input_wifi-csi']
        #csi_data = torch.mean(csi_data, dim =2)
        scale = 6
        length_test = 27
        
        csi_data = torch.tensor(csi_data)
        csi_data = csi_data.cuda()
        csi_data = csi_data.type(torch.cuda.FloatTensor)
        #csi_dafeaturesta = csi_data.view(16,2,3,114,10)
        keypoint = data['output']#17,3
        keypoint = keypoint.cuda()
       
        
        xy_keypoint = keypoint[:,:, 0:2].cuda()
        confidence = keypoint[:,:, 2:3].cuda()
        
        #add noise and denoise
        
        csi_data = csi_data.to('cpu').numpy()
        noise = epsilon*np.random.normal(0, 1, [length_test,3,114,10])
        #csi_data = csi_data + noise
        csi_data = add_salt_and_pepper_noise(csi_data, noise_level=0.02)
        
        numtaps = 30  # Số lượng hệ số bộ lọc
        cutoff = 0.1  # Tần số cắt-off của bộ lọc
        b = firwin(numtaps, cutoff)


        #for i in range(csi_data.shape[0]):
        #    for j in range(csi_data.shape[1]):
        #        for k in range(csi_data.shape[2]):
        #            csi_data[i, j, k, :] = medfilt(csi_data[i, j, k, :], kernel_size = 5)
        
        
        
        csi_data = torch.tensor(csi_data)
        csi_data = csi_data.to('cuda')
        csi_data = csi_data.type(torch.cuda.FloatTensor)

        pred_xy_keypoint,time = metafi(csi_data) #b,2,17,17
        pred_xy_keypoint = pred_xy_keypoint.squeeze()
        #pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 1, 2)
        pred_xy_keypoint = pred_xy_keypoint.reshape(length_test,17,2)
        
        #print('time: %.3f' % time)
        
        #loss = criterion_L2(torch.mul(pred_xy_keypoint, label))/32
        loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))
        test_loss_iter.append(loss.cpu().detach().numpy())

        #optimizer.zero_grad()

        #loss.backward()
        #optimizer.step()

        #lr = np.array(scheduler.get_last_lr())
        pred_xy_keypoint = pred_xy_keypoint.cpu()
        xy_keypoint = xy_keypoint.cpu()
        pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
        xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)
       
        pck = compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
        
        metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))

        pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck , 0.5))
        pck_40_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.4))
        pck_30_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.3))
        pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck , 0.2))
        pck_10_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.1))
        pck_5_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.05))  
        
        
   
    test_mean_loss = np.mean(test_loss_iter)
    sum_time = np.sum(time_iter)
    mean = np.mean(metric, 0)*1000
    mpjpe_mean = mean[0]
    pa_mpjpe_mean = mean[1]
    pck_50 = np.mean(pck_50_iter, 0)
    pck_40 = np.mean(pck_40_iter, 0)
    pck_30 = np.mean(pck_30_iter, 0)
    pck_20 = np.mean(pck_20_iter, 0)
    pck_10 = np.mean(pck_10_iter, 0)
    pck_5 = np.mean(pck_5_iter, 0)
    pck_50_overall = pck_50[17]
    pck_40_overall = pck_40[17]
    pck_30_overall = pck_30[17]
    pck_20_overall = pck_20[17]
    pck_10_overall = pck_10[17]
    pck_5_overall = pck_5[17]

    print('test result with loss: %.3f, pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, pck_20: %.3f, pck_10: %.3f, pck_5: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f' % (test_mean_loss, pck_50_overall,pck_40_overall, pck_30_overall,pck_20_overall, pck_10_overall,pck_5_overall, mpjpe_mean, pa_mpjpe_mean))
    print('-----pck_50-----')
    print(pck_50)
    print('-----pck_40-----')
    print(pck_40)
    print('-----pck_30-----')
    print(pck_30)
    print('-----pck_20-----')
    print(pck_20)
    print('-----pck_10-----')
    print(pck_10)
    print('-----pck_5-----')
    print(pck_5)    






