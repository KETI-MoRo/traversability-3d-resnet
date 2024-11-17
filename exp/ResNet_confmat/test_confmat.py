import argparse

import os
import numpy as np
from datetime import datetime

import torch.nn.functional as F
import torch
import torch.nn as nn
import pytorchvideo
import pytorchvideo.models.resnet
from torchvision import transforms

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


# calculate confusion matrix
def compute_confmat(logits, labels):
    pred = torch.argmax(logits, dim=1)
    TP = torch.zeros(logits.shape[1])  # True Positives
    FP = torch.zeros(logits.shape[1])  # False Positives
    FN = torch.zeros(logits.shape[1])  # False Negatives

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if pred[i] == j and labels[i] == j:
                TP[j] += 1
            elif pred[i] == j and labels[i] != j:
                FP[j] += 1
            elif pred[i] != j and labels[i] == j:
                FN[j] += 1

    return TP, FP, FN


def evaluate(model, val_loader, fp_):
    model.eval()
    loss_meter = Meter()
    acc_meter = Meter()

    for i, (data, labels) in enumerate(val_loader):
        data = data.cuda().permute(0,2,1,3,4)
        labels = labels.cuda()
        logits = model(data)

        loss = F.cross_entropy(logits, labels)
        acc = compute_accuracy(logits, labels)
        TP_, FP_, FN_ = compute_confmat(logits, labels)

        # Accumulate TP, FP, FN
        if i == 0 :
            TP = TP_
            FP = FP_
            FN = FN_
        else :
            TP += TP_
            FP += FP_
            FN += FN_

        # Output class probabilities and labels to text file
        if False:
            logits_ = logits.cpu().detach().numpy()
            labels_ = labels.cpu().detach().numpy()
            for batch_idx in range(logits_.shape[0]):
                logits_softmax = np.exp(logits_[batch_idx]) / np.sum(np.exp(logits_[batch_idx]))

                # Output to text file
                fp_.write('{} {} {} {} {} {}\n'.format(logits_softmax[0],
                                            logits_softmax[1],
                                            logits_softmax[2],
                                            labels_[batch_idx],
                                            i, batch_idx))

        loss_meter.update(loss.item())
        acc_meter.update(acc)

        # Print progress
        if i % 10 == 0 :
            print('{}th data is evaluated.'.format(i))

    # Calculate precision, recall, f1_scores.
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    #return loss_meter.avg(), acc_meter.avg(), precision, recall, f1_scores
    return loss_meter.avg(), acc_meter.avg(), precision.tolist(), recall.tolist(), f1_scores.tolist()


def main(args):

    # Creation of output folder.
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)
    
    # Initialize the output textfile.
    current_time = datetime.now().strftime("%y%m%d-%H%M%S")
    fp = open(os.path.join(args.path_output, 'test_{}.txt'.format(current_time)),'w')
    #fp.close()

    # print the model information
    print('model : {}'.format( args.model ))
    fp.write('model : {}\n'.format( args.model ))

    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(224),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(224),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_val = VideoFrameDataset(
        root_path=os.path.join(args.path_input,'CURB2023'),
        annotationfile_path=args.path_annotationfile,
        num_segments=16,
        frames_per_segment=1,
        transform=preprocess,
        test_mode=False
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size = 2,
        shuffle=False,
        num_workers=2,
        pin_memory = True,
    )

    # Model Creation
    model = make_resnet()
    model = model.cuda()
    load_model(model, args.model)

    # Test
    val_loss, val_acc, precision, recall, f1_scores = evaluate(model, val_loader, fp)
    print('val/loss : {}, val/acc : {}'.format( val_loss, val_acc ))
    fp.write('val/loss : {}, val/acc : {}\n'.format( val_loss, val_acc ))
    print('precision : {}, recall : {}, f1_scores : {}'.format( precision, recall, f1_scores ))
    fp.write('precision : {}, recall : {}, f1_scores : {}\n'.format( precision, recall, f1_scores ))

    fp.close()


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='exp_ResNet_confmat/test.py', add_help=add_help)

    parser.add_argument('--path_input', default= './dataset', type=str, help=('path to dataset (CURB2023)'))

    parser.add_argument('--path_output', default= './output/exp_ResNet_confmat', type=str, help=('path to output folder'))

    parser.add_argument('--model', default='./output/train/240512-060023/fold0/max_acc.tar', type=str)
    parser.add_argument('--path_annotationfile', default='./output/train/240512-060023/fold0/annotation_val.txt', type=str)
    # parser.add_argument('--model', default='./output/train/240512-060023/fold1/max_acc.tar', type=str)
    # parser.add_argument('--path_annotationfile', default='./output/train/240512-060023/fold1/annotation_val.txt', type=str)
    # parser.add_argument('--model', default='./output/train/240512-060023/fold2/max_acc.tar', type=str)
    # parser.add_argument('--path_annotationfile', default='./output/train/240512-060023/fold2/annotation_val.txt', type=str)
    # parser.add_argument('--model', default='./output/train/240512-060023/fold3/max_acc.tar', type=str)
    # parser.add_argument('--path_annotationfile', default='./output/train/240512-060023/fold3/annotation_val.txt', type=str)
    # parser.add_argument('--model', default='./output/train/240512-060023/fold4/max_acc.tar', type=str)
    # parser.add_argument('--path_annotationfile', default='./output/train/240512-060023/fold4/annotation_val.txt', type=str)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
