import argparse
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import pytorchvideo
import pytorchvideo.models.resnet
from torchvision import transforms
from torchvision import utils as vutils
import tqdm
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))), 'src'))
from video_dataset import  VideoFrameDataset, ImglistToTensor
from utils import compute_accuracy, Meter, load_model


def make_resnet():
    return pytorchvideo.models.resnet.create_resnet(
        input_channel = 3,
        model_depth = 50,
        model_num_class = 3,
        norm=nn.BatchNorm3d,
        activation = nn.ReLU,
    )


def evaluate(epoch, model, val_loader, args):
    model.eval()
    loss_meter = Meter()
    acc_meter = Meter()
    tqdm_gen = tqdm.tqdm(val_loader)

    for i, (data, labels) in enumerate(tqdm_gen):
        data = data.cuda().permute(0,2,1,3,4)
        labels = labels.cuda()
        logits = model(data)

        # Restore permuted tensor for CAM output
        data = data.permute(0,2,1,3,4)

        loss = F.cross_entropy(logits, labels)
        acc = compute_accuracy(logits, labels)

        logits_ = logits.cpu().detach().numpy()
        labels_ = labels.cpu().detach().numpy()
        for batch_idx in range(logits_.shape[0]):
            logits_softmax = np.exp(logits_[batch_idx]) / np.sum(np.exp(logits_[batch_idx]))

            # Output text file.
            fp = open(os.path.join(args.path_output, 'test.txt'),'a')
            fp.write('{} {} {} {} {} {}\n'.format(logits_softmax[0],
                                          logits_softmax[1],
                                          logits_softmax[2],
                                          labels_[batch_idx],
                                          i, batch_idx))
            fp.close()

        #'''
        # Compute CAM (Class Activation Map)
        #   [NOTE!!] The output attribute of `model.blocks[4].output` is added by code modification.
        #       Details are described in README.md.
        weights = model.blocks[5].proj.weight  # weights.shape : torch.Size([3, 2048])
        conv_output = model.blocks[4].output    # conv_output.shape : torch.Size([1, 2048, 16, 7, 7])

        cam = []

        for class_idx in range(weights.shape[0]) : # weights.shape[0] : 3

            cam.append([])
            for sub_array_idx in range(conv_output.shape[2]) : # 16
                cam[class_idx].append([])

            for weight_idx in range(weights.shape[1]) : # weights.shape[1] : 2048

                weight = weights[class_idx, weight_idx]

                sub_array = conv_output[0,weight_idx,:,:,:] # 16, 7, 7

                for sub_array_idx in range(sub_array.shape[0]) : # sub_array.shape[0]) : 16
                
                    # Extract a single frame from sub_array
                    sub_array_frame = sub_array[sub_array_idx,:,:]

                    if weight_idx == 0 :
                        cam[class_idx][sub_array_idx] = weight * sub_array_frame
                    else :
                        cam[class_idx][sub_array_idx] = cam[class_idx][sub_array_idx] + weight * sub_array_frame

            for sub_array_idx in range(conv_output.shape[2]) : # 16

                # Apply ReLU
                cam_output = F.relu(cam[class_idx][sub_array_idx])  # Apply ReLU to get positive values

                cam_output = cam_output.unsqueeze(0).unsqueeze(0)  # Reshape CAM for visualization

                # Upsample CAM to match original frame size
                upsample = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                cam_upsampled = upsample(cam_output.squeeze()).squeeze()

                # Extract rgb image
                #rgb = data[0,sub_array_idx,:,:,:]
                rgb = load_original_frame( i, sub_array_idx, args )
                rgb = upsample(rgb).permute(1,2,0)
                
                # Save to file
                plt.imshow(rgb)
                plt.imshow(cam_upsampled, cmap="jet", alpha=0.5)
                plt.axis('off')
                plt.savefig('./output/exp_CAM/data{}_class{}_frame{}.png'.format(i, class_idx, sub_array_idx),
                            bbox_inches="tight", pad_inches = 0)
                #'''


        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[val] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


# Load original RGB for CAM overlay
def load_original_frame( sequence_idx, frame_idx, args ):

    #annotationfile_path = os.path.join(os.path.dirname(args.path_weight), 'annotation_val.txt')
    annotationfile_path = './exp/CAM/list_CAM.txt'

    f = open(annotationfile_path)
    lines = f.readlines()
    line_idx = 0
    for line in lines:
        line = line.strip().split()[0]

        if sequence_idx == line_idx :
            break;

        line_idx = line_idx + 1

    f.close()

    # path to original RGB
    frame_path = '{}/CURB2023/{}/{}_{:06d}.jpg'.format(args.path_input, line, line, frame_idx*2)
    #print(frame_path)

    frame_rgb = Image.open(frame_path)

    process = transforms.Compose([
        #ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(224),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(224),  # image batch, center crop to square 299x299
        transforms.ToTensor()
    ])

    frame_rgb = process(frame_rgb)

    return frame_rgb


#========================================
# main function
def main(args):

    # Creation of output folder.
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)
    
    # Initialize the output textfile.
    fp = open(os.path.join(args.path_output, 'test.txt'),'w')
    fp.close()

    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(224),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(224),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_val = VideoFrameDataset(
        root_path=os.path.join(args.path_input,'CURB2023'),
        annotationfile_path=os.path.join(os.path.dirname(args.path_weight), 'annotation_val.txt'),
        num_segments=16,
        frames_per_segment=1,
        transform=preprocess,
        test_mode=False
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size = 1,
        shuffle=False,
        num_workers=2,
        pin_memory = True,
    )

    model = make_resnet()
    model = model.cuda()

    load_model(model, args.path_weight)

    for epoch in range(1):
        val_loss, val_acc, _ =  evaluate(epoch, model, val_loader, args)
        print('epoch : {}, val/loss : {}, val/acc : {}'.format( epoch, val_loss, val_acc))


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description='test.py', add_help=add_help)

    parser.add_argument('--path_input', default= './dataset', type=str, help=('path to dataset (CURB2023)'))

    parser.add_argument('--path_output', default= './output/exp_CAM', type=str, help=('path to output folder'))

    parser.add_argument('--path_weight', default= './output/train/240512-060023/fold2/max_acc.tar', type=str, help=('path to weight file'))    

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
