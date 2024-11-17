import argparse
import os
from datetime import datetime

# import python files in the src folder
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

# Bayesian Temporal Fusion
from ClassConfidenceOfSequence import ClassConfidence


# Calculate the starting point of the first frame (image) in each sublist
def get_sublist_start_idx(args) :

    idx_starts = [0]
    if args.sublist_len >= 2 :
        idx_starts.append(args.sublist_frames_num - args.video_length + 1)
    if args.sublist_len >= 3 :
        for i in range(2,args.sublist_len):
            idx_starts.append(idx_starts[1]+(i-1)*args.sublist_frames_num)

    return idx_starts


# Calculate the number of true positives
def calc_true_positive_num(args, test_file, f_4wheel, f_6wheel, f_prohibition, margin) :

    # Calculate the starting point of the first frame (image) in each sublist
    idx_starts = get_sublist_start_idx(args)

    # Retrieve the label values
    path_test_file = os.path.join(args.path_permutation, test_file)
    fp = open(path_test_file, 'r')
    lines = fp.readlines()
    fp.close()

    assert(len(lines) == args.sublist_len)
    
    labels = []
    for line in lines :
        line = line.strip()  # Remove newline characters
        label = line.split('_')[-1]
        labels.append(label)    # Retrieve the label values

    true_positive_num = 0
    false_positive_num = 0
    false_negative_num = 0
    confusion_matrix = ''   # confusion matrix value

    # Check at the starting point of the first frame (image) in each sublist
    for i in range(1, len(idx_starts)) :    # Perform tests only at the change points between sublists, so 1 is excluded

        # If the sublist changes and recognition occurs within the margin, it is considered a true positive
        for j in range(margin + 1) :

            idx = idx_starts[i] + j

            # Combine Bayesian fusion results corresponding to idx into a single list
            f = [ f_4wheel[idx], f_6wheel[idx], f_prohibition[idx] ]
            f_str = [ '4wheel', '6wheel', 'prohibition' ]

            # Find the maximum value among the Bayesian fusion results
            f_max_idx = f.index(max(f))

            # TP (True Positive)
            if labels[i] == f_str[f_max_idx] :

                # set confusion matrix value
                confusion_matrix = 'TP'

                # Stop checking
                break

            # FN (False Negative)
            elif labels[i-1] == f_str[f_max_idx] :

                # set confusion matrix value
                confusion_matrix = 'FN'

            # FP (False Negative)
            else :

                # set confusion matrix value
                confusion_matrix = 'FP'

        # Process confusion matrix values
        if confusion_matrix == 'TP' :
            true_positive_num = true_positive_num + 1
        elif confusion_matrix == 'FN' :
            false_negative_num = false_negative_num + 1
        else :
            false_positive_num = false_positive_num + 1
    
    return true_positive_num, false_positive_num, false_negative_num


#========================================
# main func
def main(args):

    # Create output folder.
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    # Save output as a text file
    current_time = datetime.now().strftime("%y%m%d-%H%M%S")
    fp_precision = open(os.path.join(args.path_output,'temporal_fusion_{}.txt'.format(current_time)), 'w')

    # path to the ResNet 3D inference output of permutation frames
    print('test(inference) data path : {}'.format(args.path_test_output))
    fp_precision.write('test(inference) data path : {}\n'.format(args.path_test_output))
    
    # Calculate the starting point of the first frame (image) in each sublist
    idx_starts = get_sublist_start_idx(args)
    fp_precision.write('start idx of {} sublists : {}\n'.format(args.sublist_len, idx_starts))
    print('start idx of {} sublists : {}'.format(args.sublist_len, idx_starts))

    # Output files of the ResNet 3D inference (test)
    test_files = sorted(os.listdir(args.path_test_output))

    # Due to the large number of permutation.txt files, permutation_frames output files may be processed in parts.
    test_files = test_files[ args.idx_start : (args.idx_end + 1) ]

    print('Number of test permutations from {} to {} : {}'.format(
        args.idx_start, 
        args.idx_end, 
        len(test_files)))
    fp_precision.write('Number of test permutations from {} to {} : {}\n'.format(
        args.idx_start, 
        args.idx_end, 
        len(test_files)))

    # Output the margin to a text file
    fp_precision.write('args.true_false_margins : {}\n'.format(args.true_false_margin))
    print('args.true_false_margin : {}'.format(args.true_false_margin))

    # The for loop to apply various margins
    for margin in range(5, args.true_false_margin + 1) :

        true_positive_num = 0
        false_positive_num = 0
        false_negative_num = 0
        test_num = 0

        # The for loop to process test result files
        for test_file in test_files :

            # Open the inference (test) result file
            path_test_file = os.path.join(args.path_test_output, test_file)
            fp = open(path_test_file, 'r')
            lines = fp.readlines()
            fp.close()

            # Verify if the test result file contains a sufficient number of lines
            len_lines = args.sublist_frames_num * args.sublist_len - args.video_length + 1
            assert(len(lines) == len_lines), '{}'.format(path_test_file)

            p_4wheel = []
            p_6wheel = []
            p_prohibition = []

            # Copy data from the text file into each list
            for line in lines :

                data = line.split(' ')

                p_4wheel.append(float(data[0]))
                p_6wheel.append(float(data[1]))
                p_prohibition.append(float(data[2]))

            # Python class for Bayesian fusion's class confidence
            class_num = 3   # number of class
            class_confidence = ClassConfidence(class_num)
            f_4wheel = []
            f_6wheel = []
            f_prohibition = []
            
            # Execute Bayesian temporal fusion
            for i in range(len(p_4wheel)) :

                # Calculate class confidence for temporal decision fusion(Bayesian temporal fusion)
                f_t = class_confidence.evaluate([p_4wheel[i], p_6wheel[i], p_prohibition[i]])

                # Save class confidence
                f_4wheel.append(f_t[0])
                f_6wheel.append(f_t[1])
                f_prohibition.append(f_t[2])

            # Calculate the number of true positives
            tp_num, fp_num, fn_num = calc_true_positive_num(args, test_file, f_4wheel, f_6wheel, f_prohibition, margin)
            true_positive_num = true_positive_num + tp_num
            false_positive_num = false_positive_num + fp_num
            false_negative_num = false_negative_num + fn_num
            test_num = test_num + (args.sublist_len - 1)    # Perform tests only at the change points between sublists, so 1 must be subtracted

        # precision and Recall
        str_output = 'margin : {:2d}, Num of TP, FP, FN : {}, {}, {}, Num of test : {}, Precision : {}, Recall : {}'.format(
            margin,
            true_positive_num, false_positive_num, false_negative_num,
            test_num,
            true_positive_num/(true_positive_num + false_positive_num), # Precision
            true_positive_num/(true_positive_num + false_negative_num)) # Recall, sensitivity
        
        # Output Precision and Recall results to a text file
        fp_precision.write(str_output + '\n')
        print(str_output)
        
    # Close the text file
    fp_precision.close()


def get_args_parser(add_help=True):

    description = 'Calculate the Bayesian temporal fusion of test.py output.'
    parser = argparse.ArgumentParser(description=description, add_help=add_help)

    # output folder for temporal fusion
    parser.add_argument('--path_output', type=str,
                        default= './output/exp_mode_adaptation/temporal_fusion/',
                        help=('output folder for temporal fusion'))

    # path to the ResNet 3D inference output of synthetic sequences
    parser.add_argument('--path_test_output', type=str,
        default= './output/exp_mode_adaptation/test/',
        help=('path to the ResNet 3D inference output of synthetic sequences'))

    # path to the output of permutation.py
    parser.add_argument('--path_permutation', type=str,
        default= './output/exp_mode_adaptation/permutation/241117-051723/',
        help=('path to the output of permutation.py'))

    # Length (number of sequences) of the sublist generated by permutations
    parser.add_argument('--sublist_len', type=int,
        default= 4,
        help=('Length (number of sequences) of the sublist generated by permutations'))

    # The frames number of a video sequece of the input dataset (CURB2023)
    # [NOTE] `sublist_frames_num` may not be identical to `video_length`.
    parser.add_argument('--sublist_frames_num', type=int,
        default= 32,
        help=('The frames number of a video sequece of the input dataset (CURB2023)'))

    # Length of the video sequence (i.e., number of images) input to test.py (ResNet 3D inference)
    parser.add_argument('--video_length', type=int,
        default= 16,
        help=('Length of the video sequence (i.e., number of images) input to test.py'))

    # If the sublist changes and recognition occurs within this margin, it is considered a true positive
    parser.add_argument('--true_false_margin', type=int,
        #default= 5, # minimum of margin
        default= 20,
        #default= 32,   # maximum of margin
        help=('If the sublist changes and recognition occurs within this margin, it is considered a true positive'))
    
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
