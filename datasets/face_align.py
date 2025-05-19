import os
import cv2
import insightface
from insightface.utils import face_align
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser(description='face alignment')
parser.add_argument('-i', '--input_dir', default='Asian_celebrity', help='input image directory')
parser.add_argument('-o', '--output_dir', default='Asian_celebrity_align', help='output image directory')
args = parser.parse_args()

# PATH
input_folder = os.path.abspath(args.input_dir)
output_folder = os.path.abspath(args.output_dir)  
os.makedirs(output_folder, exist_ok=True)

# Infer face detection by CPU
app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider', "CUDAExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640)) 

for root, _, files in os.walk(input_folder): 
    for image_name in files:
        image_path = os.path.join(root, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"{image_name} - 이미지 로드 실패")
            continue

        # 저장 경로 구성
        relative_path = os.path.relpath(root, input_folder)
        save_folder = os.path.join(output_folder, relative_path)
        output_path = os.path.join(save_folder, image_name)

        # 이미 만들어진 것이 있으면 추가로 하지 않음
        if os.path.exists(output_path):
            print(f"{image_name} 이미 처리됨, 건너뜀")
            continue

        # 얼굴 검출
        faces = app.get(img)
        if len(faces) == 0:
            print(f"{image_name}에서 얼굴 검출 실패")
            continue

        # 얼굴 정렬
        face = faces[0]
        aligned_face = face_align.norm_crop(img, face.kps)

        os.makedirs(save_folder, exist_ok=True)
        cv2.imwrite(output_path, aligned_face)
        print(f"{image_name} 정렬 완료")