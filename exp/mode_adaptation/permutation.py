import argparse
import os
from datetime import datetime
import itertools    # permutation, combination


def convert_valid_list_to_dictionary(path_valid):

    f = open(path_valid, 'r')
    lines = f.readlines()
    f.close()
    print('The validation list is loaded.')

    list_dict_valid = []

    for line in lines:
        line = line.strip()
        #print(line)
        line_split = line.split(' ')
        name = line_split[0]
        label = name.split('_')[-1]

        # Create a dictionary and store it in a list
        dict_valid = dict([('name', name), ('label', label)])
        list_dict_valid.append(dict_valid)
    
    print('Converting to dictionary is completed.')

    return list_dict_valid


#========================================
# main func
def main(args):

    print(args.path_validation_list)
    
    current_time = datetime.now().strftime("%y%m%d-%H%M%S")
    path_output_folder = os.path.join(args.path_output,current_time)

    # Create the output folder if it does not exist
    if not os.path.exists(path_output_folder):
        os.makedirs(path_output_folder)
        print('{} is created.\n'.format(path_output_folder))
    
    # Prepare a log file
    fp = open(os.path.join(args.path_output,'permutation_{}.txt'.format(current_time)),'w')
    fp.write('This is a log file for permutation.py.\n\n')

    # Load the list of video sequences from the validation dataset list.
    list_dict_valid = convert_valid_list_to_dictionary(args.path_validation_list)
    print('{} is loaded.\n'.format(args.path_validation_list))
    fp.write('{} is loaded.\n'.format(args.path_validation_list))

    # Generate sublists of length args.sublist_len using permutations.
    # [NOTE] Sublists may include consecutive labels of the same type.
    permutations = list(itertools.permutations(list_dict_valid, args.sublist_len))

    print('len of permutations : {}'.format(len(permutations)))
    fp.write('len of permutations : {}\n'.format(len(permutations)))

    # Check if two consecutive elements in each sublist have the same label, and exclude such sublists
    permutations2 = []

    for perm in permutations:
        labels = [elem['label'] for elem in perm]
        if not any(labels[i] == labels[i + 1] for i in range(len(labels) - 1)):
            #print([elem['name'] for elem in perm])
            permutations2.append(perm)

    print('len of permutations2 : {}'.format(len(permutations2)))
    fp.write('len of permutations2 : {}\n'.format(len(permutations2)))

    # Output sublists to text files
    idx = 0
    for perm in permutations2:

        # Output file name
        filename = os.path.join(args.path_output, current_time, 'permutation{0:010d}.txt'.format(idx))
        print(filename)

        # Retrieve video names as a list
        names = [elem['name'] for elem in perm]

        f = open(filename, 'w')
        for name in names:            
            f.write('{}\n'.format(name))
        f.close()

        idx = idx + 1

    fp.write('Saving txt files is completed.\n')
    fp.close()


def get_args_parser(add_help=True):
    
    parser = argparse.ArgumentParser(description='Make list of permutations of validation dataset videos.', add_help=add_help)

    # validation video sequence list
    parser.add_argument('--path_validation_list', type=str,
        default= './output/train/240512-060023/fold2/annotation_val.txt',
        help=('list of video sequences not utilized in the training phase'))

    # Length (number of sequences) of the sublist generated by permutations
    parser.add_argument('--sublist_len', type=int,
        default= 4,
        help=('Length (number of sequences) of the sublist generated by permutations'))

    # Folder path to save the sublist as a text file
    parser.add_argument('--path_output', type=str,
        default= './output/exp_mode_adaptation/permutation/',
        help=('Folder path to save the sublist as a text file'))
    
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
