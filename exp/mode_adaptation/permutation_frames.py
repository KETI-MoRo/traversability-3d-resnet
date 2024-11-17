import argparse
import os
from datetime import datetime


# Retrieve video names from permutation.txt in sequential order
def load_permutation( path, idx):
    
    # permutation.txt file name
    filename = os.path.join(path, 'permutation{0:010d}.txt'.format(idx))

    f = open(filename, 'r')
    lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]	# Remove newline characters
    f.close()

    # [NOTE] Do not sort as it is a permutation!
    return lines


# Retrieve image names based on the video names stored in the permutation.txt
def load_image_names( path, permutation):

    images = []

    for i in range(len(permutation)):

        path_folder = os.path.join( path, permutation[i])

        folder_images = sorted(os.listdir(path_folder))

        # Prepend the folder path to the image file names
        folder_images = [os.path.join(path_folder,folder_image) for folder_image in folder_images]

        images = images + folder_images

    # `images` is a list of paths to image files
    return images


# Output `the list of images for inference` to a text file
def output_frames_list( idx_permutation, images, path_output, video_length):

    # Create path_output folder
    if not os.path.exists(path_output) :
        os.makedirs(path_output)

    # Create permutation{0:05d} folder in the path_output folder
    # Create a folder for each permutation.txt file
    # This folder is referred to as the `permutation folder`.
    foldername = os.path.join(path_output,'permutation{0:010d}'.format(idx_permutation))
    if not os.path.exists(foldername) :
        os.mkdir(foldername)

    # In `permutation folder`, output frame{0:05d}.txt.
    for i in range(len(images) - video_length + 1) :

        # Open a file
        filename = os.path.join(foldername,'frame{0:03d}.txt'.format(i))        
        fp = open(filename,'w')

        # Output paths to frame images, with the number of paths equal to video_length
        for j in range(video_length) :
            fp.write('{}\n'.format(images[i+j]))

        # Close the file
        fp.close()


#========================================
# main 함수
def main(args):

    current_time = datetime.now().strftime("%y%m%d-%H%M%S")
    path_output_folder = os.path.join(args.path_output,current_time)

    print('start idx : {}, end idx : {}'.format(
        args.idx_start, args.idx_end))
    
    # Due to the large number of permutation.txt files, permutation_frames are processed in parts.
    for idx_permutation in range( args.idx_start, args.idx_end + 1):

        # Retrieve video names from permutation.txt in sequential order
        permutation = load_permutation(args.path_permutation, idx_permutation)

        # Retrieve image names based on the video names stored in the permutation.txt
        images = load_image_names(args.path_videos, permutation)

        #print('len of images : {}'.format(len(images)))

        # Output `the list of images for inference` to a text file
        output_frames_list( idx_permutation, images, path_output_folder, args.video_length)


def get_args_parser(add_help=True):

    description = 'Generate lists of image frames from the results of permutation.py for input to test.py'
    parser = argparse.ArgumentParser(description=description, add_help=add_help)

    # output folder path for lists of image frames from the results of permutation.py
    parser.add_argument('--path_output', type=str,
        default= './output/exp_mode_adaptation/permutation_frames/',
        help=('output folder path for lists of image frames from the results of permutation.py'))

    # path to the output of permutation.py
    parser.add_argument('--path_permutation', type=str,
        default= './output/exp_mode_adaptation/permutation/241117-051723/',
        help=('path to the output of permutation.py'))

    # path to video sequence (jpg file folders)
    parser.add_argument('--path_videos', type=str,
        default= './dataset/CURB2023/',
        help=('path to video sequence (jpg file folders)'))

    # Length of the video sequence (i.e., number of images) input to test.py
    parser.add_argument('--video_length', type=int,
        default= 16,
        help=('Length of the video sequence (i.e., number of images) input to test.py'))
    
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
