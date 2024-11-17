import argparse
import matplotlib.pyplot as plt
import os

# import python files in the src folder
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

from ClassConfidenceOfSequence import ClassConfidence


#========================================
# main func
def main(args):

    print(args.path_test_output_txt)

    # Open the input text file
    fp = open(args.path_test_output_txt, 'r')
    lines = fp.readlines()
    fp.close()

    p_4wheel = []
    p_6wheel = []
    p_prohibition = []

    # Copy the input text file data to each lists
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


    #====================================
    # Graph plot
    
    # (matplotlib) Graph size setting
    plt_fig = plt.figure()
    plt_nrow = 3
    plt_ncol = 1
    plt_width = 960 * plt_ncol / plt_fig.dpi   # subplot 2열 크기 지정
    plt_height = 400 * plt_nrow / plt_fig.dpi   # subplot 2행 크기 지정
    plt_fig.set_figwidth(plt_width)
    plt_fig.set_figheight(plt_height)

    # (matplotlib) Subplot setting #1
    # https://m.blog.naver.com/jung2381187/220408468960
    plt_ax1 = plt_fig.add_subplot( plt_nrow, plt_ncol, 1)
    plt_ax1.set_title('Changes in probability over sequence step')
    #plt_ax1.set_xlim([0, len(p_4wheel)])
    plt_ax1.set_xlim([-10, len(p_4wheel) + 10])
    plt_ax1.set_xlabel('Sequence step')
    ax_ylim = 1.05
    plt_ax1.set_ylim([0, ax_ylim])
    plt_ax1.set_ylabel('Probability')
    plt_ax1.grid(True)

    # (matplotlib) Plot text file data
    # https://wikidocs.net/92085
    plt_ax1.plot(p_4wheel,\
        color='green', marker='o', linestyle='-.', label='3D ResNet (4wheel)')
    plt_ax1.plot(p_6wheel,\
        color='blue', marker='+', linestyle='-.', label='3D ResNet (6wheel)')
    plt_ax1.plot(p_prohibition,\
        color='red', marker='*', linestyle='-.', label='3D ResNet (prohibition)')
    plt_ax1.legend(loc='center right')

    # (matplotlib) Subplot setting #2
    # https://m.blog.naver.com/jung2381187/220408468960
    plt_ax2 = plt_fig.add_subplot( plt_nrow, plt_ncol, 2)
    #plt_ax2.set_title('Changes in probability over sequence step')
    #plt_ax2.set_xlim([0, len(p_4wheel)])
    plt_ax2.set_xlim([-10, len(p_4wheel) + 10])
    plt_ax2.set_xlabel('Sequence step')
    ax_ylim = 1.05
    plt_ax2.set_ylim([0, ax_ylim])
    plt_ax2.set_ylabel('Probability')
    plt_ax2.grid(True)

    # (matplotlib) Plot text file data
    # https://wikidocs.net/92085
    plt_ax2.plot(f_4wheel,\
        color='limegreen', marker='o', linestyle='-', label='Bayesian (4wheel)')
    plt_ax2.plot(f_6wheel,\
        color='cyan', marker='+', linestyle='-', label='Bayesian (6wheel)')
    plt_ax2.plot(f_prohibition,\
        color='lightcoral', marker='*', linestyle='-', label='Bayesian (prohibition)')
    plt_ax2.legend(loc='center right')

    # (matplotlib) Subplot setting #3
    # https://m.blog.naver.com/jung2381187/220408468960
    plt_ax3 = plt_fig.add_subplot( plt_nrow, plt_ncol, 3)
    #plt_ax3.set_title('Changes in probability over sequence step')
    #plt_ax3.set_xlim([0, len(p_4wheel)])
    plt_ax3.set_xlim([-10, len(p_4wheel) + 10])
    plt_ax3.set_xlabel('Sequence step')
    ax_ylim = 1.05
    plt_ax3.set_ylim([0, ax_ylim])
    plt_ax3.set_ylabel('Probability')
    plt_ax3.grid(True)

    # (matplotlib) Plot text file data
    # https://wikidocs.net/92085
    plt_ax3.plot(p_4wheel,\
        color='green', marker='o', linestyle='-.', label='3D ResNet (4wheel)')
    plt_ax3.plot(f_4wheel,\
        color='limegreen', marker='o', linestyle='-', label='Bayesian (4wheel)')
    plt_ax3.plot(p_6wheel,\
        color='blue', marker='+', linestyle='-.', label='3D ResNet (6wheel)')
    plt_ax3.plot(f_6wheel,\
        color='cyan', marker='+', linestyle='-', label='Bayesian (6wheel)')
    plt_ax3.plot(p_prohibition,\
        color='red', marker='*', linestyle='-.', label='3D ResNet (prohibition)')
    plt_ax3.plot(f_prohibition,\
        color='lightcoral', marker='*', linestyle='-', label='Bayesian (prohibition)')
    plt_ax3.legend(loc='center right')

    # (matplotlib) Save the figure as a PNG file
    timestamp = args.path_test_output_txt.split('/')[-2]
    foldername = os.path.join(args.path_output,timestamp)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filename = args.path_test_output_txt.split('/')[-1]
    plt.savefig(os.path.join(foldername,'{}.png'.format(filename.split('.')[0])))

    print('{} is saved.'.format(
        os.path.join(foldername,'{}.png'.format(filename.split('.')[0]))
    ))


def get_args_parser(add_help=True):

    description = 'Visualize a synthetic sequence.'
    parser = argparse.ArgumentParser(description=description, add_help=add_help)

    # path to the visualizatoin input (a text file including ResNet 3D output of a synthetic sequence)
    parser.add_argument('--path_test_output_txt', type=str,
        default= './output/exp_mode_adaptation/test/241117-080850/permutation0000000200.txt',
        help=('path to the visualizatoin input (a text file including ResNet 3D output of a synthetic sequence'))
    
    # path to the output folder
    parser.add_argument('--path_output', type=str,
        default= './output/exp_mode_adaptation/visualize_a_permutation/',
        help=('path to the output folder'))
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)