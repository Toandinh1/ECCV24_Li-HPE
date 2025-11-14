# HPE-Li 
This repo is the official implementation for [HPE-Li: WiFi-enabled Lightweight Dual Selective Kernel Convolution for Human Pose Estimation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04496.pdf), which was published at ECCV 2024. You can check the extension version in the TAI branch. This is a version of HPE-Li's extension, which is published in the IEEE Transactions on Artificial Intelligence journal (named HPE-Li++). This repository allows researchers and practitioners to reproduce our results on MM-Fi and WiPose datasets.

## Data Preparation
### Download datasets.

#### There are 2 datasets to download:

- MM-Fi Dataset
- WiPose Dataset

#### MM-Fi Dataset

1. Request dataset [here](https://ntu-aiot-lab.github.io/mm-fi)
2. Download the WiFi datasets:
   1. `MMFI_Dataset.zip`
   2. `MMFI_action_segments.csv`
3. Unzip all files from `MMFI_Dataset.zip` to `./data/mmfi/dataset` following directory structure:

```
- data/
  - mmfi/
    - dataset/
      - E01/
        - S01/
          - A01/
            - rgb/
            - mmwave/
            - wifi-csi/
              ...
```

#### WiPose Dataset

1. Request dataset [here](https://github.com/NjtechCVLab/Wi-PoseDataset)
2. Unzip all files from `Wi-Pose.rar` to `./data/wipose/` following directory structure:

```
- data/
  - wipose/
    - Train/
    - Test/
              ...
```

