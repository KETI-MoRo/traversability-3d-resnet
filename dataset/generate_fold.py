import argparse
import os
import shutil
import random


# main 함수
def main(args):

    # 전체 sequence 목록의 경로
    path_4wheel = 'sequence_4wheel.txt'
    path_6wheel = 'sequence_6wheel.txt'
    path_prohibition = 'sequence_prohibition.txt'
    path_output ='./'

    # 터미널에서 실행하는 것이 아니라, 
    # vscode에서 `generate_fold.py`를 상위폴더에서 실행하는 경우를
    # 대비하기 위한 path 수정
    if not os.path.exists(path_4wheel) :
        path_output ='./dataset'
        path_4wheel = os.path.join(path_output, path_4wheel)
        path_6wheel = os.path.join(path_output, path_6wheel)
        path_prohibition = os.path.join(path_output, path_prohibition)
    
    # output 디렉토리 내의 모든 폴더 목록 가져오기
    folders = sorted(os.listdir(path_output))

    # output 디렉토리 내의 "fold"로 시작하는 폴더 삭제
    for folder in folders:
        if folder.startswith("fold"):
            folder_path = os.path.join(path_output, folder)
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder}")
            except Exception as e:
                print(f"Error deleting folder {folder}: {e}")

    # 전체 sequence 목록을 불러들입니다.
    f = open(path_4wheel, 'r')
    lines_4wheel = f.readlines()
    f.close()
    f = open(path_6wheel, 'r')
    lines_6wheel = f.readlines()
    f.close()
    f = open(path_prohibition, 'r')
    lines_prohibition = f.readlines()
    f.close()

    # fold의 개수에 따라서 랜덤하게 추출할 시퀀스 개수를 계산합니다.
    valid_len_4wheel = int(len(lines_4wheel) / args.fold_number)
    valid_len_6wheel = int(len(lines_6wheel) / args.fold_number)
    valid_len_prohibition = int(len(lines_prohibition) / args.fold_number)
    
    # fold의 개수만큼 반복합니다.
    for fold in range(args.fold_number) :

        # 랜덤하게 validation sequence 추출
        lines_valid_4wheel = random.sample(lines_4wheel, valid_len_4wheel)
        lines_valid_6wheel = random.sample(lines_6wheel, valid_len_6wheel)
        lines_valid_prohibition = random.sample(lines_prohibition, valid_len_prohibition)

        # 추출된 sequence를 제외한 나머지를 training sequence로 추출
        lines_train_4wheel = [element for element in lines_4wheel if element not in lines_valid_4wheel]
        lines_train_6wheel = [element for element in lines_6wheel if element not in lines_valid_6wheel]
        lines_train_prohibition = [element for element in lines_prohibition if element not in lines_valid_prohibition]

        # 출력 텍스트파일 경로 생성
        os.mkdir(os.path.join(path_output,'fold{}'.format(fold)))
        path_valid = os.path.join(path_output,'fold{}/annotation_val.txt'.format(fold))
        path_train = os.path.join(path_output,'fold{}/annotation_train.txt'.format(fold))

        # validation sequence 텍스트파일 출력
        f = open( path_valid, 'w')

        for i in range(len(lines_valid_4wheel)):
            f.write(lines_valid_4wheel[i])

        for i in range(len(lines_valid_6wheel)):
            f.write(lines_valid_6wheel[i])

        for i in range(len(lines_valid_prohibition)):
            f.write(lines_valid_prohibition[i])

        f.close()

        # training sequence 텍스트파일 출력
        f = open( path_train, 'w')

        for i in range(len(lines_train_4wheel)):
            f.write(lines_train_4wheel[i])

        for i in range(len(lines_train_6wheel)):
            f.write(lines_train_6wheel[i])

        for i in range(len(lines_train_prohibition)):
            f.write(lines_train_prohibition[i])

        f.close()

        print('Lists of training and validation sequences of fold{} are saved.'.format(fold))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate text files for x-fold cross validation.')
    parser.add_argument('--fold_number', default=5, type=int, help='number of folds for cross validation')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args=parse_args()

    main(args)
