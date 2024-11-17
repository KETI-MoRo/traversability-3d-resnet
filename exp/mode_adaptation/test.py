import os
import argparse
import shutil
from datetime import datetime

import torch.nn.functional as F
import torch
import torch.nn as nn
import pytorchvideo
import pytorchvideo.models.resnet
from torchvision import transforms
import tqdm
import numpy as np

# import python files in the src folder
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
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


def evaluate(model, val_loader, fp):
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

            # Output text file.
            fp.write('{} {} {} {} {} {}\n'.format(logits_softmax[0],
                                          logits_softmax[1],
                                          logits_softmax[2],
                                          labels_[batch_idx],
                                          i, batch_idx))

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        #tqdm_gen.set_description(f'[val] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


# In order to share the dataloader, the dataset files are copied to a temporary directory according to permutaiton_frame.txt.
def create_test_frames_folders(args, path_frames):

    # Create a temporary folder
    path_tmp = os.path.join(args.path_output,'tmp')
    if os.path.exists(path_tmp):
        shutil.rmtree(path_tmp)
    os.makedirs(path_tmp)

    # Crate a temporary annotation_test.txt in the temporary folder
    fp_annotation = open(os.path.join(args.path_output,'tmp','annotation_test.txt'),'w')

    # Get video list from path_frames
    videos = sorted(os.listdir(path_frames))
    #print(videos[0])   # frame000.txt

    for video in videos :

        # Create a folder for each video
        path_video = os.path.join(path_tmp,video.split('.')[0])
        os.mkdir(path_video)

        # File output on annotation.txt
        # [NOTE] The label data in annotation.txt is ignored in our experimental scheme.
        fp_annotation.write('{} 1 16 0\n'.format(video.split('.')[0]))

        # Read the image list of video
        fp = open(os.path.join(path_frames,video),'r')
        lines = fp.readlines()
        fp.close()

        idx = 0

        for line in lines :
            line = line.rstrip('\n')    # Remove line breaks
            #print(line)    # ./dataset/CURB2023/2022-07-18-10-42-55_frame4275_4wheel/2022-07-18-10-42-55_frame4275_4wheel_000000.jpg

            # Path to the image file for destination
            path_dst = os.path.join(path_video,'image{0:02d}.{1}'.format(idx,line.split('.')[-1]))
            #print(path_dst)

            # Copy the image file
            shutil.copyfile(line, path_dst)

            idx = idx + 1
    
    # File close for annotation_test.txt
    fp_annotation.close()
    

#========================================
# main func
def main(args):

    current_time = datetime.now().strftime("%y%m%d-%H%M%S")
    path_output_folder = os.path.join(args.path_output,current_time)

    # Create output folder.
    if not os.path.exists(path_output_folder):
        os.makedirs(path_output_folder)

    print('start idx : {}, end idx : {}'.format(
        args.idx_start, args.idx_end))

    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(224),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(224),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = make_resnet()
    model = model.cuda()

    load_model(model, args.path_weight)
    
    # Due to the large number of permutation.txt files, test(inference) are processed in parts.
    for idx in range( args.idx_start, args.idx_end + 1):

        # If the idx is out of permutation index in permutation_frames, test(inference) is skipped.
        path_frames = os.path.join(args.path_permutation_frames,'permutation{0:010d}'.format(idx))
        if not os.path.exists(path_frames) :
            break

        # Initialize output text file.
        filename = 'permutation{0:010d}.txt'.format(idx)
        fp = open(os.path.join(path_output_folder,filename),'w')
    
        # In order to share the dataloader, the dataset files are copied to a temporary directory according to permutaiton_frame.txt.
        create_test_frames_folders(args, path_frames)

        dataset_val = VideoFrameDataset(
            root_path= os.path.join(args.path_output,'tmp'),
            annotationfile_path= os.path.join(args.path_output,'tmp','annotation_test.txt'),
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

        # inference
        val_loss, val_acc, _ =  evaluate(model, val_loader, fp)
        print('idx : {}, val/loss : {}, val/acc : {}'.format( idx, val_loss, val_acc))
                
        fp.close()


def get_args_parser(add_help=True):

    description = 'Test(inference) of frame.txt'
    parser = argparse.ArgumentParser(description=description, add_help=add_help)

    # output folder for ResNet 3D inference output
    parser.add_argument('--path_output', type=str,
                        default= './output/exp_mode_adaptation/test/',
                        help=('output folder for inference output'))
    
    # weight file for inference
    parser.add_argument('--path_weight',
                        default= './output/train/240512-060023/fold2/max_acc.tar', type=str,
                        help=('path to weight file')) 
    
    # path to permutation_frames
    parser.add_argument('--path_permutation_frames',
                        default= './output/exp_mode_adaptation/permutation_frames/241117-063417/', type=str,
                        help=('path to permutation_frames')) 
    
    # Due to the large number of permutation.txt files, permutation_frames are processed in parts.
    # Starting index for `permutation_frames.py` process.
    parser.add_argument('--idx_start', type=int,
                      default= 0,
                      help=('Starting index for `permutation_frames.py` process.'))
    
    # Due to the large number of permutation.txt files, permutation_frames are processed in parts.
    # Ending index for `permutation_frames.py` process.
    parser.add_argument('--idx_end', type=int,
                      default= 299,
                      #default= 2999,    # The Setting for IEEE Access
                      #default= 425447, # permutation0000425447.txt
                      help=('Ending index for `permutation_frames.py` process.'))
    
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
