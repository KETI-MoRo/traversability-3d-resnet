import argparse
import os
from datetime import datetime

# import python files in the src folder
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

# Bayesian Temporal Fusion
from ClassConfidenceOfSequence import ClassConfidence


# 각 sublist의 첫번째 프레임(이미지)가 시작되는 시점 계산
def get_sublist_start_idx(args) :

    idx_starts = [0]
    #idx_starts = []
    if args.sublist_len >= 2 :
        idx_starts.append(args.sublist_frames_num - args.video_length + 1)
    if args.sublist_len >= 3 :
        for i in range(2,args.sublist_len):
            idx_starts.append(idx_starts[1]+(i-1)*args.sublist_frames_num)

    return idx_starts


# True positive 개수를 계산
def calc_true_positive_num(args, test_file, f_4wheel, f_6wheel, f_prohibition, margin) :

    # 각 sublist의 첫번째 프레임(이미지)가 시작되는 시점 계산
    idx_starts = get_sublist_start_idx(args)
    #print(idx_starts)

    # label값을 갖고 옵니다.
    path_test_file = os.path.join(args.path_permutation, test_file)
    fp = open(path_test_file, 'r')
    lines = fp.readlines()
    fp.close()

    #print('len of label data : {}'.format(len(lines)))
    assert(len(lines) == args.sublist_len)
    
    labels = []
    for line in lines :
        line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
        #print('{}'.format(line))
        label = line.split('_')[-1]
        #print('{}'.format(label))
        labels.append(label)    # label값을 갖고 옵니다.

    true_positive_num = 0
    false_positive_num = 0
    false_negative_num = 0
    confusion_matrix = ''   # confusion matrix 값

    # 각 sublist의 첫번째 프레임(이미지)가 시작되는 시점마다 확인
    for i in range(1, len(idx_starts)) :    # sublist 사이의 변경시점에서만 test를 수행하므로 1을 빼야 함

        # sublist가 바뀌고 이 margin 안에 인식이 되면, true positive로 판단
        #for j in range(args.true_false_margin + 1) :
        for j in range(margin + 1) :    # 다양한 margin을 적용하기 위한 for문 적용할 때 수정됨

            # 테스트
            #print('calc_true_positive_num $ i of sublist : {}, j of margin : {}'.format(i,j))

            idx = idx_starts[i] + j

            # idx에 해당하는 Bayesian fusion 결과값을 하나의 리스트로 합치기
            f = [ f_4wheel[idx], f_6wheel[idx], f_prohibition[idx] ]
            f_str = [ '4wheel', '6wheel', 'prohibition' ]

            # Bayesian fusion 결과값 중 최대값 찾기
            f_max_idx = f.index(max(f))

            # TP (True Positive, 긍정 정답) 라면
            if labels[i] == f_str[f_max_idx] :

                # confusion matrix 값
                confusion_matrix = 'TP'

                # 확인 중지
                break

            # FN (False Negative, 부정 오류, 미인식) 이라면
            elif labels[i-1] == f_str[f_max_idx] :

                # confusion matrix 값
                confusion_matrix = 'FN'

            # FP (False Negative, 긍정 오류, 오인식) 이라면
            else :

                # confusion matrix 값
                confusion_matrix = 'FP'

        # confusion matrix 값 처리
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

    # test(inference) 결과가 저장된 경로
    print('test(inference) data path : {}'.format(args.path_test_output))
    fp_precision.write('test(inference) data path : {}\n'.format(args.path_test_output))
    
    # 각 sublist의 첫번째 프레임(이미지)가 시작되는 시점 계산
    idx_starts = get_sublist_start_idx(args)
    fp_precision.write('start idx of {} sublists : {}\n'.format(args.sublist_len, idx_starts))
    print('start idx of {} sublists : {}'.format(args.sublist_len, idx_starts))

    # test 결과파일들
    test_files = sorted(os.listdir(args.path_test_output))

    # 데이터가 너무 많아서, 나눠서 테스트 결과를 처리
    test_files = test_files[ args.idx_start : (args.idx_end + 1) ]

    print('Number of test permutations from {} to {} : {}'.format(
        args.idx_start, 
        args.idx_end, 
        len(test_files)))
    fp_precision.write('Number of test permutations from {} to {} : {}\n'.format(
        args.idx_start, 
        args.idx_end, 
        len(test_files)))

    # sublist가 바뀌고 이 margin 안에 인식이 되면, true positive로 판단하는 margin을 텍스트파일로 출력
    fp_precision.write('args.true_false_margins : {}\n'.format(args.true_false_margin))
    print('args.true_false_margin : {}'.format(args.true_false_margin))


    # 다양한 margin을 적용하기 위한 for문
    for margin in range(5, args.true_false_margin + 1) :

        true_positive_num = 0
        false_positive_num = 0
        false_negative_num = 0
        test_num = 0

        # test 결과파일에 대한 for문
        for test_file in test_files :

            # test 결과 파일 열기
            path_test_file = os.path.join(args.path_test_output, test_file)
            fp = open(path_test_file, 'r')
            lines = fp.readlines()
            fp.close()

            # test 결과 파일이 충분한 line 수를 갖고 있는지 검증
            len_lines = args.sublist_frames_num * args.sublist_len - args.video_length + 1
            assert(len(lines) == len_lines), '{}'.format(path_test_file)    # test 결과 파일 검증

            # 데이터가 저장될 리스트
            p_4wheel = []
            p_6wheel = []
            p_prohibition = []

            # 텍스트파일 데이터를 리스트로 복사
            for line in lines :

                data = line.split(' ')

                p_4wheel.append(float(data[0]))
                p_6wheel.append(float(data[1]))
                p_prohibition.append(float(data[2]))

            # Bayesian fusion의 class confidence를 위한 클래스
            class_num = 3   # class 개수
            class_confidence = ClassConfidence(class_num)
            f_4wheel = []
            f_6wheel = []
            f_prohibition = []
            
            # Bayesian fusion 수행
            for i in range(len(p_4wheel)) :

                #print('i :{}'.format(i))

                # temporal decision fusion의 class confidence를 계산합니다.
                f_t = class_confidence.evaluate([p_4wheel[i], p_6wheel[i], p_prohibition[i]])

                # class confidence를 저장합니다.
                f_4wheel.append(f_t[0])
                f_6wheel.append(f_t[1])
                f_prohibition.append(f_t[2])

            # True positive 개수를 계산
            tp_num, fp_num, fn_num = calc_true_positive_num(args, test_file, f_4wheel, f_6wheel, f_prohibition, margin)
            true_positive_num = true_positive_num + tp_num
            false_positive_num = false_positive_num + fp_num
            false_negative_num = false_negative_num + fn_num
            test_num = test_num + (args.sublist_len - 1)    # sublist 사이의 변경시점에서만 test를 수행하므로 1을 빼야 함

        # precision 결과
        str_output = 'margin : {:2d}, Num of TP, FP, FN : {}, {}, {}, Num of test : {}, Precision : {}, Recall : {}'.format(
            margin,
            true_positive_num, false_positive_num, false_negative_num,
            test_num,
            true_positive_num/(true_positive_num + false_positive_num), # Precision, 오인식이 얼마나 적나?
            true_positive_num/(true_positive_num + false_negative_num)) # Recall, sensitivity, 미인식이 얼마나 적나?
        
        # precision 결과 텍스트파일로 출력
        fp_precision.write(str_output + '\n')
        print(str_output)
        
    # 파일 닫기
    fp_precision.close()


def get_args_parser(add_help=True):

    description = 'Calculate the Bayesian temporal fusion of test.py output.'
    parser = argparse.ArgumentParser(description=description, add_help=add_help)

    # output folder for temporal fusion
    parser.add_argument('--path_output', type=str,
                        default= './output/exp_mode_adaptation/temporal_fusion/',
                        help=('output folder for temporal fusion'))

    # path to the ResNet 3D inference output of permutation frames
    parser.add_argument('--path_test_output', type=str,
        default= './output/exp_mode_adaptation/test/',
        help=('path to the ResNet 3D inference output of permutation frames'))

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
