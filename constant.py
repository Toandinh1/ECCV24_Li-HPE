#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:16:35 2024

@author: jackson-devworks
"""

experiment_config = {
    "mmfi_config": "/home/jackson-devworks/Desktop/ECCV_2024/dataset_lib/config.yaml",
    "dataset_root": "/home/jackson-devworks/Desktop/HPE/Dataset",
    "noise_level": [0.0],
    "mode": 0,  # Mode 0: no denoiser layer, Mode 1: have AE denoiser layers, Mode 2: use traditional filter to denoise
    "epoch": 60,
    "checkpoint": "/home/jackson-devworks/Desktop/ECCV_2024/output",
}

denoiser_config = {
    "epoch": 20,
    "mode": 1,  # Mode 0: 1 stage AE, Mode 1: stacked AE
    "previous_encoder": "/home/jackson-devworks/Desktop/ECCV_2024/output/SPN/FourLayerDenosing/Encoder-DecoderReconstructor",
    "checkpoint": "/home/jackson-devworks/Desktop/ECCV_2024/output/AWGN/FiveLayerDenosing/Encoder-DecoderReconstructor",
}
