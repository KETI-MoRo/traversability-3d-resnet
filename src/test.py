import argparse
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import pytorchvideo
import pytorchvideo.models.resnet
from torchvision import transforms
from video_dataset import  VideoFrameDataset, ImglistToTensor
import tqdm
from utils import compute_accuracy, Meter, load_model
import numpy as np
import time

from datetime import datetime


def make_resnet():
    return pytorchvideo.models.resnet.create_resnet(
        input_channel = 3,
        model_depth = 50,
        model_num_class = 3,
        norm=nn.BatchNorm3d,
        activation = nn.ReLU,
    )


def evaluate(epoch, model, val_loader, path_output_folder):
    model.eval()
    loss_meter = Meter()
    acc_meter = Meter()
    tqdm_gen = tqdm.tqdm(val_loader)

    for i, (data, labels) in enumerate(tqdm_gen):
        data = data.cuda().permute(0,2,1,3,4)
        labels = labels.cuda()
        logits = model(data)

        loss = F.cross_entropy(logits, labels)
        acc = compute_accuracy(logits, labels)

        logits_ = logits.cpu().detach().numpy()
        labels_ = labels.cpu().detach().numpy()
        for batch_idx in range(logits_.shape[0]):
            logits_softmax = np.exp(logits_[batch_idx]) / np.sum(np.exp(logits_[batch_idx]))

            max_index = np.argmax(logits_softmax)

            # Output text file.
            fp = open(os.path.join(path_output_folder,'test.txt'),'a')
            fp.write('{} {} {} {} {} {} {}\n'.format(logits_softmax[0],
                                          logits_softmax[1],
                                          logits_softmax[2],
                                          max_index,
                                          labels_[batch_idx],
                                          i, batch_idx))
            fp.close()

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[val] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


#========================================
# main function
def main(args):

    current_time = datetime.now().strftime("%y%m%d-%H%M%S")

    # Creation of output folder.
    path_output_folder = os.path.join(args.path_output,current_time)
    if not os.path.exists(path_output_folder):
        os.makedirs(path_output_folder)

    # Initialize the output textfile.
    fp = open(os.path.join(path_output_folder,'test.txt'),'w')
    fp.close()
    
    print('Time check starts.')
    start = time.time()

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
        batch_size = 2,
        shuffle=False,
        num_workers=2,
        pin_memory = True,
    )


    model = make_resnet()
    model = model.cuda()

    load_model(model, args.path_weight)

    for epoch in range(1):
        val_loss, val_acc, _ =  evaluate(epoch, model, val_loader, path_output_folder)
        print('epoch : {}, val/loss : {}, val/acc : {}'.format( epoch, val_loss, val_acc))
    
    end = time.time()
    print(f"{end - start:.5f} sec")
    fp = open(os.path.join(path_output_folder,'test.txt'),'a')
    fp.write('time elapsed : {:.5f} sec'.format(end - start))
    fp.close()


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description='test.py', add_help=add_help)

    parser.add_argument('--path_input', default= './dataset', type=str, help=('path to dataset (CURB2023)'))

    parser.add_argument('--path_output', default= './output/test', type=str, help=('path to output folder'))

    parser.add_argument('--path_weight', default= './output/train/240512-060023/fold2/max_acc.tar', type=str, help=('path to weight file'))    

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
