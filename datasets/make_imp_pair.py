import os
import shutil
import random
import argparse


parser = argparse.ArgumentParser(description='face alignment')
parser.add_argument('-i', '--input_dir', default='Asian_celebrity_align', 
                    help='input image directory')
parser.add_argument('-o', '--output_dir', default='Asian_celebrity_pair', 
                    help='final image directory')
parser.add_argument('-n', '--num_pairs', type=int, default=3000,
                    help='number of pairs to generate')
args = parser.parse_args()

# Path
input_dir = os.path.abspath(args.input_dir)
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)  # output 폴더 생성
output_folder = "imp"
output_path = os.path.join(output_dir, output_folder)
os.makedirs(output_path, exist_ok=True) # imposter 폴더 생성

# Initialize
male_dirs = []
female_dirs = []
person_images = {}

# 사람별 이미지 리스트 저장 및 성별 구분
for person_folder in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person_folder)

    # 디렉토리가 아니면 pass
    if not os.path.isdir(person_path):
        continue

    # 각 디렉토리 내 이미지 파일(jpg, png) 가져오기
    image_files = [f for f in os.listdir(person_path) if f.endswith('.jpg') or f.endswith('.png')]
    if not image_files:
        continue

    # split 통해 성별 확인
    gender = image_files[0].split('_')[-1].split('.')[0]
    person_images[person_folder] = image_files

    if gender == 'm':
        male_dirs.append(person_folder)
    elif gender == 'w':
        female_dirs.append(person_folder)

print(f"Male folders num: {len(male_dirs)} / smaple: {male_dirs[:5]}")
print(f"Female folders num: {len(female_dirs)} / sample: {female_dirs[:5]}")

# 쌍 만들기 전 체크
gender_groups = []
if len(male_dirs) >= 2:
    gender_groups.append(male_dirs)
if len(female_dirs) >= 2:
    gender_groups.append(female_dirs)

pair_list = []

if not gender_groups:
    pass
else:
    while len(pair_list) < args.num_pairs:
        # 성별을 랜덤으로 선택
        gender_group = random.choice(gender_groups)

        # 해당 성별 내 남은 사람이 2명 이상이 아니면 제거
        if len(gender_group) < 2:
            gender_groups.remove(gender_group)
            continue

        # 랜덤으로 2명 선택
        person1, person2 = random.sample(gender_group, 2)
        img1 = random.choice(person_images[person1])
        img2 = random.choice(person_images[person2])

        pair_list.append((person1, img1, person2, img2))

print(f"생성된 pair_list num: {len(pair_list)} / smaple: {pair_list[:5]}")

# copy selected pairs into directory
for idx, (person1, img1, person2, img2) in enumerate(pair_list):
    pair_folder = os.path.join(output_path, str(idx))
    os.makedirs(pair_folder, exist_ok=True)

    source1 = os.path.join(input_dir, person1, img1)
    source2 = os.path.join(input_dir, person2, img2)
    dest1 = os.path.join(pair_folder, f"pair1_{img1}")
    dest2 = os.path.join(pair_folder, f"pair2_{img2}")

    shutil.copy(source1, dest1)
    shutil.copy(source2, dest2)

print(f"Pair {idx} 복사 완료, 마지막 샘플: {dest1}, {dest2}")
