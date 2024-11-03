# Traversability 3D ResNet

This is the official PyTorch implementation for "Mode Prediction and Adaptation for a Six-Wheeled Mobile Robot Capable of Curb-Crossing in Urban Environments".

## Installation

### Docker Container
- [pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel](https://hub.docker.com/layers/pytorch/pytorch/1.11.0-cuda11.3-cudnn8-devel/images/sha256-9bfcfa72b6b244c1fbfa24864eec97fb29cfafc065999e9a9ba913fa1e690a02?context=explore)
- An example of Ubuntu command for creation of the Docker container.
    - Launches the Docker container in interactive mode (`-it`).
    - Enables GPU support (`--gpus all`).
    - Names the container `traversability-3d-resnet`.
    - Sets the working directory inside the container to `/workspace` (`-w /workspace`), so when you enter the container, you’ll be in the `/workspace` folder by default.
    ```bash
    $ sudo docker run -it --gpus all --name traversability-3d-resnet -w /workspace pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
    ```
- Entering the `traversability-3d-resnet` Container
    ```bash
    $ sudo docker exec -it traversability-3d-resnet /bin/bash
    ```

### Git Clone
- Installing the Git in the container
    ```bash
    # apt-get update
    # apt-get install git
    ```
- Setting your Git `user.name` and `user.email`.
    - `[NOTE]` Replace <USER_NAME> and <USER_EMAIL> with your GitHub account data.
    ```bash
    # git config --global user.name "<YOUR_NAME>"
    # git config --global user.email <USER_EMAIL>
    ```
- Cloning our Git repository.
    ```bash
    # cd /workspace
    # git clone https://github.com/KETI-MoRo/traversability-3d-resnet.git
    ```

### requirements.txt
- Install every required packages.
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # pip install -r requirements.txt
    ```

## Dataset Preparation
- This section will be updated.
- The folder tree
    '''
    nvadmin@gpu1:/mnt/nvme1n1p1/sword32/workspace2/traversability-3d-resnet$ tree -d -L 2 ./
    ./
    ├── dataset
    │   ├── CURB2023
    │   │   ├── [video name 0]
    │   │   │   ├── [video name 0]_000000.jpg
    │   │   │   ├── [video name 0]_000001.jpg
    │   │   │   ├── ...
    │   │   │   └── [video name 0]_000031.jpg
    │   │   ├── ...
    │   │   └── ...
    │   ├── fold0
    │   ├── fold1
    │   ├── fold2
    │   ├── fold3
    │   └── fold4
    ├── exp
    │   ├── CAM
    │   ├── hough
    │   ├── moving_windows
    │   ├── ResNet_confmat
    │   ├── six_videos
    │   └── six_videos_statistics
    └── src
    '''

## Preparation for 5-Fold Cross-Validation
- The dataset is randomly divided into 5 folds.
- To ensure reproducibility for training and testing, the random splits are saved as CSV files.
- Although the Python code to create these CSV files is provided (see below), we recommend using the CSV files in this repository to maintain consistency with our 5-fold setup.
- (Optional) Command to generate the CSV files:
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # /opt/conda/bin/python /workspace/traversability-3d-resnet/dataset/generate_fold.py
    ```

## 3D ResNet Train
- train.py
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # /opt/conda/bin/python /workspace/traversability-3d-resnet/src/train.py
    ```

## 3D ResNet Test
- test.py
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # /opt/conda/bin/python /workspace/traversability-3d-resnet/src/test.py --path_weight ./output/train/240512-060023/fold2/max_acc.tar
    ```

## (Experiment) CAM (Class Activation Map)
- We modified forward() of PyTorchVideo `resnet.py` for generation the CAM.
    - Install nano on our Docker container.
        ```bash
        # apt install nano -y
        ```
    - Edit `resnet.py`.
        ```bash
        # nano -c /opt/conda/lib/python3.8/site-packages/pytorchvideo/models/resnet.py
        ```
    - Find `class ResStage(nn.Module)`
        - In my `resnet.py`, `class ResStage(nn.Module)` is on `line 1362/1395`.
    - Add the code `self.output = x`. (Almost, the line is at the end of `resnet.py`.)
        ```Python
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _, res_block in enumerate(self.res_blocks):
                x = res_block(x)

            self.output = x    # Add this line.

            return x
        ```
    - Write out in the Nano editor.    
- test_CAM.py
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # /opt/conda/bin/python /workspace/traversability-3d-resnet/exp/CAM/test_CAM.py --path_weight ./output/train/240512-060023/fold2/max_acc.tar
    ```