#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:09:24 2024

@author: toangian
"""

0#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:17:34 2023

@author: toangian
"""
import xxsubtype
import torchvision
import torch.nn as nn
import torch
import cv2
from torchvision.transforms import Resize
from minirocket_3variables import fit, transform
from ChannelTrans import ChannelTransformer
from resblock import ResidualBlock
from model import Conv2D, regression, BiLSTMModel,WiPose_benchmark
from sklearn.base import BaseEstimator, ClassifierMixin
#from DynamicConv import Dynamic_conv2d, BasicBlock
from SK_network import SKConv, SKUnit
import time
import torch.nn.functional as F
from SENet import SEBasicBlock

class posenet(nn.Module):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """
    def __init__(self):
        super(posenet, self).__init__()
        
        k1= 4 #number branches of DyConv1
        k2= 4 #number branches of DyConv2
        num_lay = 64#numer hidden dim of DyConv1
        D = 64 #number hidden dim of BiLSTM
        N = 1 #number hidden layers of BiLSTm 
        R = 32 # Reduction Ratios
        T = 64 # Temperature
        hidden_reg = 32 #number hidden dim of Regression
        self.tf = ChannelTransformer(vis=True, img_size=[17, 2], channel_num=num_lay*2, num_layers=1, num_heads=3)
        self.rb1 = ResidualBlock(in_channels =1 , out_channels= 1, stride=1)
        self.rb2 = ResidualBlock(in_channels =1, out_channels= 1, stride =1)
        self.CNN1 = nn.Conv2d(in_channels=1, out_channels=num_lay, kernel_size=7, stride=1)
        self.CNN2 = nn.Conv2d(in_channels=num_lay, out_channels=num_lay*2, kernel_size=7, stride=1)
        self.regression = regression(input_dim =25088,output_dim = 34, hidden_dim= hidden_reg)
        #self.regression = regression(input_dim= 34*2*num_lay, output_dim = 34, hidden_dim= hidden_reg)
        self.BiLSTM = BiLSTMModel(input_dim= 342, hidden_dim= D, num_layers= N, output_dim = 17*2) #good para 2-64 with 2CNN, 2-16 more better  with 3CNN
        self.SEnet1 = SEBasicBlock(inplanes = 1 , planes = num_lay)
        self.SEnet2 = SEBasicBlock(inplanes = num_lay*2 , planes = num_lay*2)
        
        #self.SKNet1 = SKNet(input = 1,features = 114, num_modules =1, h =17, w =2, nums_block_list = [2, 1, 1, 1], strides_list = [1, 1, 1, 1])
        #self.SKNet2 = SKNet(input = 96,features = 114, num_modules =1, h =17, w =2, nums_block_list = [2, 1, 1, 1], strides_list = [1, 1, 1, 1])
        
        self.skconv1 = SKConv(input_dim = 1, output_dim = num_lay, dim1 =114, dim2 =10, pool_dim = 'freq-chan',  M=3, G=1, r=16, stride=1 ,L=32)
        self.skconv2 = SKConv(input_dim = num_lay, output_dim = num_lay*2, dim1 =57, dim2 =8, pool_dim = 'freq-chan',  M=3, G=1, r=16, stride=1 ,L=32)
        
        self.skunit1 = SKUnit(in_features=3, mid_features=num_lay, out_features=num_lay, dim1 = 114,dim2 = 10,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.skunit2 = SKUnit(in_features=num_lay, mid_features=num_lay*2, out_features=num_lay*2, dim1 = 57,dim2 = 8,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
        
        
        
        self.decode = nn.Sequential(
            nn.Conv2d(num_lay*4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((17, 2))
        
        self.decode1 = nn.Sequential(
            nn.Conv2d(514, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
       
        
        self.wipose = WiPose_benchmark()
         
       


        
        self.bn3 = nn.BatchNorm1d(2)
        self.AvgPool = torch.nn.AvgPool2d(kernel_size=2)
        self.MaxPool = torch.nn.MaxPool2d(kernel_size=2)
        self.bn = nn.BatchNorm2d(num_lay)
        self.bn1 = nn.BatchNorm2d(num_lay*2)
        self.bn2 = nn.BatchNorm2d(num_lay*2+10)
        
        
        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
    
        
    
    def forward(self,x): #16,2,3,114,32
        batch = x.shape[0]
        
        
 
        time_start = time.time()
        
        " CNN-spatio"
        
       
        
        
        " Selective-Dynamic_convolution "
        m = torch.nn.AvgPool2d((2, 2))
        x = self.skunit1(x)
        x = m(x)#[32, 64, 57, 5]
        
        
        out1 = self.skunit2(x)
        #out1 = self.bn1(out1)
        #out1 = self.relu(out1)
        
        " Max-Pooling"
        
        out1 = m(out1) 

    
        x = self.regression(out1)
        x = x.reshape(batch,17,2)
        
        
        time_end = time.time()
        time_sum = time_end - time_start

        #x = torch.transpose(x, 1, 2)

        return x,time_sum

def weights_init(m):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.xavier_normal_(m.weight.data)
    #     nn.init.xavier_normal_(m.bias.data)
    # elif classname.find('BatchNorm2d') != -1:
    #     nn.init.normal_(m.weight.data, 1.0)
    #     nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
