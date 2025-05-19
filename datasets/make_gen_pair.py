import os
import shutil
import random
from itertools import combinations
import argparse


parser = argparse.ArgumentParser(description='face alignment')
parser.add_argument('-i', '--input_dir', default='Asian_celebrity_align', 
                    help='input image directory')
parser.add_argument('-o', '--output_dir', default='Asian_celebrity_pair', 
                    help='final image directory')
parser.add_argument('-n', '--num_pairs', type=int, default=3000,
                    help='number of pairs to generate')
args = parser.parse_args()


# Input Path
input_dir = os.path.abspath(args.input_dir)

# Output Path
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)  # output 폴더 생성
output_folder = "gen"
output_path = os.path.join(output_dir, output_folder)
os.makedirs(output_path, exist_ok=True)  # gen 폴더 생성

# Intialize
random.seed(42)
global_index = 0

# 모든 조합 작업 후 select 
all_pairs = [] 

# 10대 폴더 내의 각 "이름_성별" 폴더를 순회
for person_folder in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person_folder)

    # 디렉토리가 아니면 pass
    if not os.path.isdir(person_path):
        continue

    # 각 디렉토리 내 이미지 파일(jpg, png) 가져오기
    image_files = sorted([f for f in os.listdir(person_path) if f.endswith('.jpg') or f.endswith('.png')])
    if len(image_files) < 2:
        continue
    
    # 모든 조합 생성 
    pairs = [(os.path.join(person_path, img1), os.path.join(person_path, img2)) for img1, img2 in combinations(image_files,2)]
    all_pairs.extend(pairs)

# 무작위 셔플
random.shuffle(all_pairs)
print(f"총 {len(all_pairs)} 쌍의 이미지가 생성되었습니다.")
selected_pairs = all_pairs[:args.num_pairs]
print(f"총 {len(selected_pairs)} 쌍의 이미지가 선택되었습니다.")
        
# copy selected pairs into directory
for idx, (source1, source2) in enumerate(selected_pairs):
    pair_folder = os.path.join(output_path, str(idx))
    os.makedirs(pair_folder, exist_ok=True)
    
    # 파일명 앞에 pair1_와 pair2_ 접두어 추가
    dest1 = os.path.join(pair_folder, f"pair1_{os.path.basename(source1)}")
    dest2 = os.path.join(pair_folder, f"pair2_{os.path.basename(source2)}")
    
    shutil.copy(source1, dest1)
    shutil.copy(source2, dest2)
