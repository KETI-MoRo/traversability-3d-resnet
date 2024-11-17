# Traversability 3D ResNet

This is the official PyTorch implementation for "Mode Prediction and Adaptation for a Six-Wheeled Mobile Robot Capable of Curb-Crossing in Urban Environments".
- Article title: Mode Prediction and Adaptation for a Six-Wheeled Mobile Robot Capable of Curb-Crossing in Urban Environments
- Journal acronym: [ACCESS](https://ieeexplore.ieee.org/document/10744006)
- Article DOI: [10.1109/ACCESS.2024.3492012](https://doi.org/10.1109/ACCESS.2024.3492012)
- Manuscript Number: Access-2024-33278

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
    ```
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
    ```

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

## (Experiment) Confusion matrix for ResNet Only
- An example command for confusion matrix `fold0`.
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # /opt/conda/bin/python /workspace/traversability-3d-resnet/exp/ResNet_confmat/test_confmat.py --model ./output/train/240512-060023/fold0/max_acc.tar --path_annotationfile ./output/train/240512-060023/fold0/annotation_val.txt
    ```
- An example command for confusion matrix `fold2`.
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # /opt/conda/bin/python /workspace/traversability-3d-resnet/exp/ResNet_confmat/test_confmat.py --model ./output/train/240512-060023/fold2/max_acc.tar --path_annotationfile ./output/train/240512-060023/fold2/annotation_val.txt
    ```
- Outputs
    - A terminal output example
        ```
        root@2d2a6e723ba6:/workspace/traversability-3d-resnet# /opt/conda/bin/python /workspace/traversability-3d-resnet/exp/ResNet_confmat/test_confmat.py --model ./output/train/240512-060023/fold2/max_acc.tar --path_annotationfile ./output/train/240512-060023/fold2/annotation_val.txt
        model : ./output/train/240512-060023/fold2/max_acc.tar
        0th data is evaluated.
        10th data is evaluated.
        val/loss : 0.05979643389582634, val/acc : 97.22222137451172
        precision : [1.0, 0.9230769276618958, 1.0], recall : [1.0, 1.0, 0.9333333373069763], f1_scores : [1.0, 0.9600000381469727, 0.9655172228813171]
        ```
    - See the txt files in `./output/exp_ResNet_confmat/

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
- Outputs
    - See the png files in `./output/exp_CAM/`

## (Experiment) Mode Adaptation

### Problem Definition
![synthetic_sequnce](/exp/mode_adaptation/img/synthetic_sequence.png)
- The experiment involves the random selection of four video sequences not utilized in the training phase, concatenating them to form a video for assessing mode adaptation. 
- Special attention was given to ensuring that the same class does not persist continuously in these videos.
- Please read our paper.

### Synthetic Sequence Generation
- `/exp/mode_adaptation/permutation.py/`
    - `permutation.py` makes list of permutations of validation dataset videos.
        ```bash
        # cd /workspace/traversability-3d-resnet/
        # /opt/conda/bin/python /workspace/traversability-3d-resnet/exp/mode_adaptation/permutation.py --path_validation_list ./output/train/240512-060023/fold2/annotation_val.txt
        ```
        - [NOTE] Please set the `--path_validation_list` as your ResNet 3D train output path. Check your `./output/train/` folder.
    - Outputs
        - The sublists will be saved as text files in `./output/exp_mode_adaptation/permutation/`
        - A sample of `permutation.txt` (e.g. `./output/exp_mode_adaptation/permutation/241117-051723/permutation0000425447.txt`)
            ```
            2022-07-18-10-03-53_frame3813_prohibition
            2022-05-10-17-06-56_frame0491_6wheel
            2022-07-18-09-35-46_frame1106_prohibition
            2022-07-18-10-21-12_frame2749_6wheel
            ```
- `/exp/mode_adaptation/permutation_frames.py/`
    - As you see, `permutation.txt` is a list of vide sequences.
    - To input to the ResNet 3D, they need to be converted into a list of frames according to video sequences.
    - In the `./output/exp_mode_adaptation/permutation_frames/` directory, a folder named with the timestamp ("%y%m%d-%H%M%S") is created. This folder is referred to as the `output folder`.
    - In `output folder`, a folder for each permutation.txt file is created. This folder is referred to as the `permutation folder`.
    - In `permutation folder`, paths to frame images are outpt to text file, with the number of paths equal to video_length.
    - This is an implementation of image shifting along the time axis in a `permutation.txt`.
    - Execution
        ```bash
        # cd /workspace/traversability-3d-resnet/
        # /opt/conda/bin/python /workspace/traversability-3d-resnet/exp/mode_adaptation/permutation_frames.py --idx_start 0 --idx_end 299 --path_permutation ./output/exp_mode_adaptation/permutation/241117-051723/
        ```
        - [NOTE] Please set the `--path_permutation` as your `permutation.py` output path. Check your `./output/exp_mode_adaptation/permutation/` folder.
    - Outputs
        - The frame lists will be saved as text files in `./output/exp_mode_adaptation/permutation_frames/`
        - A sample of `frame.txt` (e.g. `./output/exp_mode_adaptation/permutation_frames/241117-063417/permutation0000000000/frame020.txt`)
            ```
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000020.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000021.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000022.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000023.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000024.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000025.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000026.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000027.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000028.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000029.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000030.jpg
            ./dataset/CURB2023/2022-07-18-11-14-43_frame5390_4wheel/2022-07-18-11-14-43_frame5390_4wheel_000031.jpg
            ./dataset/CURB2023/2022-07-18-10-21-12_frame3212_6wheel/2022-07-18-10-21-12_frame3212_6wheel_000000.jpg
            ./dataset/CURB2023/2022-07-18-10-21-12_frame3212_6wheel/2022-07-18-10-21-12_frame3212_6wheel_000001.jpg
            ./dataset/CURB2023/2022-07-18-10-21-12_frame3212_6wheel/2022-07-18-10-21-12_frame3212_6wheel_000002.jpg
            ./dataset/CURB2023/2022-07-18-10-21-12_frame3212_6wheel/2022-07-18-10-21-12_frame3212_6wheel_000003.jpg
            ```

### ResNet 3D Test
- `/exp/mode_adaptation/test.py/` infers the above sublists using the ResNet 3D.
- Execution
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # /opt/conda/bin/python /workspace/traversability-3d-resnet/exp/mode_adaptation/test.py --idx_start 0 --idx_end 299 --path_weight ./output/train/240512-060023/fold2/max_acc.tar --path_permutation_frames ./output/exp_mode_adaptation/permutation_frames/241117-063417/
    ```
    - [NOTE] Please set the `--path_weight` as your ResNet 3D train output path. Check your `./output/train/` folder.
    - [NOTE] Please set the `--path_permutation_frames` as your `permutation_frames.py` output path. Check your `./output/exp_mode_adaptation/permutation_frames/` folder.

### Temporal Fusion
- Execution
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # /opt/conda/bin/python /workspace/traversability-3d-resnet/exp/mode_adaptation/temporal_fusion.py --idx_start 0 --idx_end 99 --path_test_output ./output/exp_mode_adaptation/test/241117-080850/
    ```
- Output
    - The number of TP(True Positive), FP(False Positive), FN(False Negative) will be saved as a text file in `./output/exp_mode_adaptation/permutation_frames/tempora_fusion/`.
    - A sample of `temporal_fusion.txt`
        ```
        test(inference) data path : ./output/exp_mode_adaptation/test/240529-2120
        start idx of 4 sublists : [0, 17, 49, 81]
        Number of test permutations from 0 to 2999 : 3000
        args.true_false_margins : 20
        margin :  5, Num of TP, FP, FN : 521, 142, 8337, Num of test : 9000, Precision : 0.7858220211161387, Recall : 0.05881688868819147
        margin :  6, Num of TP, FP, FN : 1137, 127, 7736, Num of test : 9000, Precision : 0.8995253164556962, Recall : 0.12814155302603403
        margin :  7, Num of TP, FP, FN : 1588, 132, 7280, Num of test : 9000, Precision : 0.9232558139534883, Recall : 0.17907081641858366
        margin :  8, Num of TP, FP, FN : 2197, 222, 6581, Num of test : 9000, Precision : 0.9082265398925176, Recall : 0.2502848029163819
        margin :  9, Num of TP, FP, FN : 2636, 302, 6062, Num of test : 9000, Precision : 0.897208985704561, Recall : 0.3030581742929409
        margin : 10, Num of TP, FP, FN : 3226, 450, 5324, Num of test : 9000, Precision : 0.8775843307943417, Recall : 0.3773099415204678
        margin : 11, Num of TP, FP, FN : 3645, 511, 4844, Num of test : 9000, Precision : 0.8770452358036573, Recall : 0.4293791966073742
        margin : 12, Num of TP, FP, FN : 4765, 1069, 3166, Num of test : 9000, Precision : 0.8167637984230374, Recall : 0.600806960030261
        margin : 13, Num of TP, FP, FN : 5755, 1044, 2201, Num of test : 9000, Precision : 0.8464480070598618, Recall : 0.7233534439416792
        margin : 14, Num of TP, FP, FN : 6623, 1388, 989, Num of test : 9000, Precision : 0.8267382349269754, Recall : 0.8700735680504467
        margin : 15, Num of TP, FP, FN : 6787, 1363, 850, Num of test : 9000, Precision : 0.8327607361963191, Recall : 0.8886997512112086
        margin : 16, Num of TP, FP, FN : 6911, 1376, 713, Num of test : 9000, Precision : 0.8339567998069265, Recall : 0.906479538300105
        margin : 17, Num of TP, FP, FN : 6973, 1363, 664, Num of test : 9000, Precision : 0.8364923224568138, Recall : 0.9130548644755794
        margin : 18, Num of TP, FP, FN : 7196, 1330, 474, Num of test : 9000, Precision : 0.8440065681444991, Recall : 0.9382007822685788
        margin : 19, Num of TP, FP, FN : 7234, 1311, 455, Num of test : 9000, Precision : 0.8465769455822119, Recall : 0.9408245545584601
        margin : 20, Num of TP, FP, FN : 7234, 1311, 455, Num of test : 9000, Precision : 0.8465769455822119, Recall : 0.9408245545584601
        ```

### Visualize a Synthetic Sequence
- Execution
    ```bash
    # cd /workspace/traversability-3d-resnet/
    # /opt/conda/bin/python /workspace/traversability-3d-resnet/exp/mode_adaptation/visualize_a_synthetic_sequence.py --path_test_output_txt ./output/exp_mode_adaptation/test/241117-080850/permutation0000000200.txt
    ```
    - [NOTE] Please set the `--path_test_output_txt` as a your text file output of the ResNet 3D inference output.
- Output
    - The visualization output will be saved as a PNG file in `./output/exp_mode_adaptation/visualize_a_permutation/`.
    - the output file name is identical with `--path_test_output_txt`.
    - Please see the Fig. 11 in [our paper](https://ieeexplore.ieee.org/document/10744006).

## Citation
```bibtex
@ARTICLE{KimDY2024traversability,
  author={Kim, Dong Yeop and Kim, Tae-Keun and Kim, Keunhwan and Hwang, Jung-Hoon and Kim, Euntai},
  journal={IEEE Access}, 
  title={Mode prediction and adaptation for a six-wheeled mobile robot capable of curb-crossing in urban environments},
  year={2024},
  volume={12},
  number={},
  pages={166474-166485},
  keywords={Mobile robots;Robots;Wheels;Urban areas;Three-dimensional displays;Navigation;Cameras;Robot vision systems;Robot sensing systems;Bayes methods;Bayesian fusion;bogie suspension;curb;deep learning;mobile robot;navigation;3D ResNet},
  doi={10.1109/ACCESS.2024.3492012}}

```

## Acknowledgment
- This work was supported by Korea Evaluation Institute of Industrial Technology ([KEIT](https://www.keit.re.kr/)) grant funded by the Korea government([MOTIE](https://www.motie.go.kr/)) {(No. 20023455, Development of Cooperate Mapping, Environment Recognition and Autonomous Driving Technology for Multi Mobile Robots Operating in Large-scale Indoor Workspace), (No.20005062, Development of Artificial Intelligence Robot Autonomous Navigation Technology for Agile Movement in Crowded Space)}