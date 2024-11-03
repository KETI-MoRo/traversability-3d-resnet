import argparse
import os
import shutil
from datetime import datetime
import time

import torch.nn.functional as F
import torch
import torch.nn as nn
import pytorchvideo
import pytorchvideo.models.resnet
from torchvision import transforms
from video_dataset import  VideoFrameDataset, ImglistToTensor

from utils import compute_accuracy, Meter


def make_resnet(device_name):
    if "NVIDIA Quadro RTX 8000" in device_name :
        return pytorchvideo.models.resnet.create_resnet(
            input_channel = 3,
            model_depth = 50,
            model_num_class = 3,
            norm=nn.BatchNorm3d,
            activation = nn.ReLU,
        )
    elif "NVIDIA A100" in device_name :
        return pytorchvideo.models.resnet.create_resnet(
            input_channel = 3,
            model_depth = 50,
            model_num_class = 3,
            norm=nn.BatchNorm3d,
            activation = nn.ReLU,
        )
    else :
        # NVIDIA GeForce RTX 2080  
        return pytorchvideo.models.resnet.create_resnet(
            input_channel = 3,
            model_depth = 50,
            model_num_class = 3,
            norm=nn.BatchNorm3d,
            activation = nn.ReLU,
        )

def train(fold, epoch, model, train_loader, optimizer):
    model.train()
    loss_meter = Meter()
    acc_meter = Meter()

    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda().permute(0,2,1,3,4)
        labels = labels.cuda()
        logits = model(data)
        loss = F.cross_entropy(logits, labels)
        acc = compute_accuracy(logits, labels)

        loss_meter.update(loss.item())
        acc_meter.update(acc)

        print('[train] fold:{} | epo:{} | avg.loss:{:.4f} | avg.acc:{:.3f} (curr:{:.3f})'.format(
            fold, epoch, loss_meter.avg(), acc_meter.avg(), acc
        ))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()

def evaluate(fold, epoch, model, val_loader):
    model.eval()
    loss_meter = Meter()
    acc_meter = Meter()

    for i, (data, labels) in enumerate(val_loader):
        data = data.cuda().permute(0,2,1,3,4)
        labels = labels.cuda()
        logits = model(data)

        loss = F.cross_entropy(logits, labels)
        acc = compute_accuracy(logits, labels)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        print('[valid] fold:{} | epo:{} | avg.loss:{:.4f} | avg.acc:{:.3f} (curr:{:.3f})'.format(
            fold, epoch, loss_meter.avg(), acc_meter.avg(), acc
        ))

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


#========================================
# main function
def main(args):

    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(224),  # image batch, resize smaller edge to 224
        #transforms.Resize(232),
        transforms.CenterCrop(224),  # image batch, center crop to square 224x224
        #transforms.CenterCrop(232),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Adjust hyperparameters to optimize for the detected GPU model.
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        device_name = torch.cuda.get_device_name(0)
    else:
        print("No GPU available.")
        device_name = "NO GPU"

    if "NVIDIA A100" in device_name :
        batch_size = 8
    elif "NVIDIA Quadro RTX 8000" in device_name :
        batch_size = 8
    else :
        # NVIDIA GeForce RTX 2080
        batch_size = 2

    print('batch_size for {} : {}'.format(device_name, batch_size))

    current_time = datetime.now().strftime("%y%m%d-%H%M%S")

    # Creation of output folder.
    path_output_folder = os.path.join(args.path_output,current_time)
    if not os.path.exists(path_output_folder):
        os.makedirs(path_output_folder)

    folds = ['fold0','fold1','fold2','fold3','fold4']   # 5 fold
    #folds = ['fold2']   # For test purpose, set fold #2 only.

    for fold in folds :

        print('{} starts....'.format(fold))

        # output folder for each fold
        path_output_fold = os.path.join(args.path_output,current_time,fold)
        if not os.path.exists(path_output_fold):
            os.makedirs(path_output_fold)

        # Initialize the output textfile.
        fp = open('{}/train.txt'.format(path_output_fold),'w')
        fp.write('batch_size for {} : {}\n'.format(device_name, batch_size))

        dataset_train = VideoFrameDataset(
            root_path=os.path.join(args.path_input,'CURB2023'),
            annotationfile_path=os.path.join(args.path_input, fold, 'annotation_train.txt'),
            num_segments=16,
            frames_per_segment=1,
            transform=preprocess,
            test_mode=False
        )
        dataset_val = VideoFrameDataset(
            root_path=os.path.join(args.path_input,'CURB2023'),
            annotationfile_path=os.path.join(args.path_input, fold, 'annotation_val.txt'),
            num_segments=16,
            frames_per_segment=1,
            transform=preprocess,
            test_mode=False
        )

        # annotation file backup
        shutil.copyfile(dataset_train.annotationfile_path, 
                        os.path.join(path_output_fold, dataset_train.annotationfile_path.split('/')[-1]))
        shutil.copyfile(dataset_val.annotationfile_path, 
                        os.path.join(path_output_fold, dataset_val.annotationfile_path.split('/')[-1]))

        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size = batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory = True,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size = batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory = True,
        )

        start_time = time.time()

        model = make_resnet(device_name)
        model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= 40, verbose = True)
        max_acc = 0.0

        for epoch in range(0, args.max_epoch):
            train_loss, train_acc, _ = train(fold, epoch, model, train_loader, optimizer)
            val_loss, val_acc, _ = evaluate(fold, epoch, model, val_loader)
            
            print('fold : {}, epoch : {}/{}, train/loss : {}, train/acc : {}, val/loss : {}, val/acc :{}'.format(
                fold, epoch, args.max_epoch,
                train_loss, train_acc, val_loss, val_acc))
            fp.write('fold : {}, epoch : {}/{}, train/loss : {}, train/acc : {}, val/loss : {}, val/acc :{}\n'.format(
                fold, epoch, args.max_epoch,
                train_loss, train_acc, val_loss, val_acc))

            lr_scheduler.step()
            state = {
                'model':model.state_dict(),
                'epoch':epoch,
                'optimizer':optimizer.state_dict(),
                'lr_scheduler':lr_scheduler.state_dict()
            }

            # Save the best train output.
            if val_acc >= max_acc:
                print('*********A better model is found ({:.3f}) *********'.format(val_acc))
                fp.write('*********A better model is found ({:.3f}) *********\n'.format(val_acc))
                max_acc = val_acc
                torch.save(state, '{}/max_acc.tar'.format(path_output_fold))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('elapsed_time for {} : {}'.format(fold,elapsed_time))
        fp.write('elapsed_time for {} : {}\n'.format(fold,elapsed_time))

        # Save the final train output.
        torch.save(state, '{}/last.tar'.format(path_output_fold))

        fp.close()


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='train.py', add_help=add_help)

    parser.add_argument('--path_input', default= './dataset', type=str, help=('path to dataset (CURB2023)'))
    
    parser.add_argument('--path_output', default= './output/train', type=str, help=('path to output folder'))
    
    parser.add_argument('--max_epoch', default= 200, type=int)
    
    parser.add_argument('--num_workers', default= 2, type=int)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
