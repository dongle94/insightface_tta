import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import cv2
from copy import deepcopy
import torch.cuda
from sklearn.metrics import roc_curve, auc
import numpy as np
import math
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import argparse
from sklearn.model_selection import KFold
from scipy import interpolate
import sys
import warnings

import tent
from backbones import get_topo_model
from utils.utils_config import get_config


warnings.filterwarnings(("ignore"))
script_dir = os.path.dirname(os.path.abspath(__file__))
# 작업 디렉토리를 스크립트 위치로 변경
os.chdir(script_dir)
print("** 현재 작업 디렉토리:", os.getcwd())

TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])


class Tee:
    """콘솔과 파일에 동시에 출력하는 클래스"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()


class FaceDataset(Dataset):
    def __init__(self, path_list, transform=None):
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)
    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, img_path


def tent_get_embeddings_from_pathlist(path_list, batch_size=16):
    dataset = FaceDataset(path_list, transform=TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings_dict = {}
    adapted_embeddings_dict = {}
    with torch.no_grad():
        print(f"** Calculating embeddings for {len(path_list)} images with batch size {batch_size}")
        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            
            batch_embeds = o_model(batch_imgs, phase='infer')  # 배치 단위로 임베딩 계산
            batch_embeds = batch_embeds.cpu().numpy()  # CPU로 변환 후 numpy 배열로 변환

            adapted_output = t_model(batch_imgs)  # 배치 단위로 임베딩 계산
            adapted_embeds = adapted_output.cpu().numpy()  # CPU로 변환 후 numpy 배열로 변환

            for path, embed, a_embed in zip(batch_paths, batch_embeds, adapted_embeds):
                embeddings_dict[path] = embed
                adapted_embeddings_dict[path] = a_embed

    return embeddings_dict, adapted_embeddings_dict


# 임베딩 거리를 계산하는 함수 (코사인 유사도)
def distance(embeding1, embeding2, distance_metric = 0):
    eps = 1e-6
    if distance_metric == 0:
        dot = np.sum(np.multiply(embeding1,embeding2), axis=1)# 벡터의 내적
        norm1 = np.linalg.norm(embeding1, ord=2, axis=1)
        norm2 = np.linalg.norm(embeding2, ord=2, axis=1)
        norm = norm1 * norm2 + eps  # 0 나누기를 피하기 위해 eps 추가
        cos_similarity = dot / norm

        dist = 1 - cos_similarity # 코사인 유사도 기반 거리 계산
        # dist = np.arccos(cos_similarity) / math.pi # 코사인 유사도 기반 각도를 0~1 사의로 정규화 = 아크코사인 사용
    else:
        raise Exception("Undefined distance metirc %d" % distance_metric)
    return dist


def get_paths(dataset_dir):
    nrof_skipped_pairs =0
    path_list = []
    issame_list = []

    genuine_path = os.path.join(dataset_dir, "gen") # genuine set의 경로
    genuine_folders = sorted(os.listdir(genuine_path))# genuine 폴더의 리스트
    genuine_count = len(genuine_folders) #총 폴더 수
    genuine_split = np.array_split(genuine_folders, 10)

    imposter_path = os.path.join(dataset_dir, "imp")
    imposter_folders = sorted(os.listdir(imposter_path))
    imposter_count = len(imposter_folders)
    imposter_split = np.array_split(imposter_folders, 10)
    
    # 10 fold를 위해 각 fold에 genuine과 imposter를 10분에 1씩 할당
    for fold_idx in range(10):
        for folder in genuine_split[fold_idx]:
            folder_path = os.path.join(genuine_path, folder)
            images = sorted(os.listdir(folder_path))
            if len(images) == 2:
                path0 = os.path.join(folder_path, images[0])
                path1 = os.path.join(folder_path, images[1])
                path_list += [path0, path1]
                issame_list.append(1)
            else:
                nrof_skipped_pairs += 1

        for folder in imposter_split[fold_idx]:
            folder_path = os.path.join(imposter_path, folder)
            images = sorted(os.listdir(folder_path))
            if len(images) == 2:
                path0 = os.path.join(folder_path, images[0])
                path1 = os.path.join(folder_path, images[1])
                path_list += [path0, path1]
                issame_list.append(0)
            else:
                nrof_skipped_pairs += 1
    
    if nrof_skipped_pairs > 0:
        print(f'Skipped {nrof_skipped_pairs} image paiars')
    return path_list, issame_list


def calculate_accuracy(threshold, dist, actual_issame):
    """
    주어진 임계값(threshold)에서 예측한 동일 인물 여부와 실제 라벨을 비교하여
    true positive, false positive, accuracy 및 오류 지표를 계산합니다.
    """
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc, is_fp, is_fn


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    """
    K-Fold 교차 검증을 통해 ROC 커브(민감도, 특이도)와 Accuracy를 계산합니다.
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
 
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive = []
    is_false_negative = []
    best_thres = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # 훈련 세트에서 최적 임계값 탐색
        acc_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _, _ = calculate_accuracy(threshold, dist[test_set],
                                                                      actual_issame[test_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, _, _ = calculate_accuracy(
                threshold,
                dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(
            thresholds[best_threshold_index], 
            dist[test_set],
            actual_issame[test_set])

        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)
        best_thres.append(thresholds[best_threshold_index])

    tpr = np.mean(tprs, axis=0)
    fpr = np.mean(fprs, axis=0)
    return tpr, fpr, accuracy, is_false_positive, is_false_negative, best_thres


def calculate_val_far(threshold, dist, actual_issame):
    """
    주어진 임계값에서의 검증율(val)과 허용 오인율(FAR)을 계산합니다.
    """
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

"""
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
   
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # 훈련 세트에서 목표 FAR에 해당하는 임계값 찾기
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean
"""
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    """
    목표 FAR에 해당하는 임계값을 찾아 검증율을 계산합니다.
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)
    best_thres = []

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # 훈련 세트에서 목표 FAR에 해당하는 임계값 찾기
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])

        # far_train에서 중복 제거
        unique_far_train, unique_indices = np.unique(far_train, return_index=True)

        if len(unique_far_train) > 1 and np.max(unique_far_train) >= far_target:
            unique_thresholds = thresholds[unique_indices]  # 중복 제거된 thresholds 선택
            f = interpolate.interp1d(unique_far_train, unique_thresholds, kind='slinear', fill_value="extrapolate")
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])
        best_thres.append(threshold)

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, best_thres


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    """
    LFW 평가를 위해 임베딩을 두 그룹(각 쌍의 이미지)으로 나누고,
    ROC 및 검증 지표를 계산합니다.
    반환 값:
      - tpr: true positive rate 배열
      - fpr: false positive rate 배열
      - accuracy: 각 fold별 Accuracy
      - val: 검증율 (validation rate)
      - val_std: 검증율의 표준편차
      - far: 평균 허용 오인율 (false acceptance rate)
      - fp, fn: false positive와 false negative 배열
    """
    thresholds = np.arange(0, 2, 0.001)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn, best_lfw_thres = calculate_roc(thresholds, embeddings1, embeddings2,
                                               np.asarray(actual_issame), nrof_folds=nrof_folds,
                                               distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 2, 0.001)
    val, val_std, far, best_ijb_thres = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds,
                                      distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far, fp, fn, best_lfw_thres, best_ijb_thres


def get_topofr_model(name='r50'):
    network = name
    if name == 'r50':
        model_config = 'configs/glint360k_r50.py'
        model_path = '../../model_zoo/Glint360K_R50_TopoFR_9727.pt'
    elif name == 'r100':
        model_config = 'configs/glint360k_r100.py'
        model_path = '../../model_zoo/Glint360K_R100_TopoFR_9760.pt'
    elif name == 'r200':
        model_config = 'configs/glint360k_r200.py'
        model_path = '../../model_zoo/Glint360K_R200_TopoFR_9784.pt'
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    cfg = get_config(model_config)
    model = get_model(network, dropout=0, fp16=False, num_features=cfg.embedding_size, num_classes=cfg.num_classes)
    weight = torch.load(model_path, weights_only=True)
    model.load_state_dict(weight)
    # model = torch.nn.DataParallel(model)

    return model, model_path


def get_tented_model(model_name, tent_steps=1, episodic=False):
    orig_model, model_path = get_topofr_model(model_name)

    model = deepcopy(orig_model)
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3)  # Example optimizer
    tented_model = tent.Tent(
        model=model, 
        optimizer=optimizer, 
        steps=tent_steps, 
        episodic=episodic
    )

    # model = torch.nn.DataParallel(resnet)
    orig_model.eval().to(device)
    tented_model.eval().to(device)

    param = dict()
    param['model_path'] = model_path
    param['optimizer'] = optimizer
    param['tent_steps'] = tent_steps
    param['episodic'] = episodic
    # param['params'] = params

    return orig_model, tented_model, param


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do benchmark test')
    # parser.add_argument('--model_config', default='configs/glint360k_r50.py', help='path to load model.')
    # parser.add_argument('--model_path', default='model/Glint360K_R50_TopoFR_9727.pth', help='path to load model.')
    # parser.add_argument('--model_dir', default='model_out', help='path to load model dir.')
    # parser.add_argument('--image_path', default='lfw_testset', type=str, help='')
    # parser.add_argument('--batch_size', default=16, type=int, help='')
    # parser.add_argument('--name', default='r50', type=str, help='')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--logs', '-l', type=str, help="ex) exp/calfw.txt")
    args = parser.parse_args()
    
    # 출력 결과를 저장하기 위해 Tee 클래스 사용, 로그 파일명 설정 (원하는 경로로 수정 가능)
    log_dir = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(__file__), args.logs)))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(os.path.dirname(__file__), args.logs)
    # 콘솔과 파일에 동시 출력하도록 설정
    sys.stdout = Tee(log_file)
    
    # 이미지 폴더 경로
    # dataset_dir = args.image_path
    # print(f"** Test Dataset: {os.path.basename(dataset_dir)}")
    
    names = ['r50', 'r100', 'r200']
    tent_steps = [5, 3, 1]
    episodic = [True]
    image_paths = [                  
    ] 

    # batch_size = args.batch_size
    device = torch.device(f'cuda:{args.gpu}'if torch.cuda.is_available() else 'cpu')
    print(f'** Running on device: {device}')

    for dataset_dir in image_paths:
        print(f"** Test Dataset: {os.path.basename(dataset_dir)}")
        for name in names:
            if name == 'r200':
                batch_sizes = [8, 16, 32]      
            else: 
                batch_sizes = [16, 32, 48]  
            for tbs in batch_sizes:
                for epi in episodic:
                    for ts in tent_steps:
                        # get model
                        o_model, t_model, param = get_tented_model(name, tent_steps=ts, episodic=epi)

                        print("** --------------------------------- **")
                        print(f"** Model loaded: {param}")

                        path_list, issame_list = get_paths(dataset_dir)
                        embeddings_dict, adapted_embeddings_dict = tent_get_embeddings_from_pathlist(path_list, batch_size=tbs)
                        
                        embeddings_eval = np.array([embeddings_dict[path] for path in path_list])
                        adapted_embeddings_eval = np.array([adapted_embeddings_dict[path] for path in path_list])
                        
                        # Test 1: 원본 모델의 임베딩
                        tpr, fpr, accuracy, val, val_std, far, fp, fn, bt_lfw, bt_ijb = evaluate(embeddings_eval, issame_list, nrof_folds=10)
                        print(f"** Accuracy for each fold: {[round(a, 5) for a in accuracy]}", )
                        print(f"** Mean Accuracy: {round(np.mean(accuracy), 6)}")
                        print(f"** Best Accuracy threshold: {[round(a, 5) for a in bt_lfw]}")
                        print(f"** TPR@FPR: {round(val, 6)}, FPR: {far}")

                        # 먼저, embeddings_eval에서 쌍별로 임베딩을 분리합니다. 
                        embeddings1 = embeddings_eval[0::2]
                        embeddings2 = embeddings_eval[1::2]

                        # 전체 쌍에 대해 거리를 계산
                        dists = distance(embeddings1, embeddings2, distance_metric=0)

                        # genuine 쌍(issame_list가 1인 경우)만 선택
                        issame_array = np.array(issame_list)
                        genuine_dists = dists[issame_array == 1]
                        imposter_dists = dists[issame_array == 0]
                        
                        # genuine 쌍의 평균 거리를 계산
                        average_genuine_dist = np.mean(genuine_dists)
                        print("** Average distance for genuine pairs:", average_genuine_dist)
                        std_genuine_dists = np.std(genuine_dists)
                        print("** Standard deviation for genuine pairs:", std_genuine_dists)
                        average_imposter_dist = np.mean(imposter_dists)
                        print("** Average distance for imposter pairs:", average_imposter_dist)
                        std_imposter_dists = np.std(imposter_dists)
                        print("** Standard deviation for imposter pairs:", std_imposter_dists)

                        print(">>>>>>>>>>>>>>>>>>>>>")

                        # Test 2: 적응 모델의 임베딩
                        tpr, fpr, accuracy, val, val_std, far, fp, fn, bt_lfw, bt_ijb = evaluate(adapted_embeddings_eval, issame_list, nrof_folds=10)
                        print(f"** Adapted Accuracy for each fold: {[round(a, 5) for a in accuracy]}", )
                        print(f"** Adapted Mean Accuracy: {round(np.mean(accuracy), 6)}")
                        print(f"** Best Adapted Accuracy threshold: {[round(a, 5) for a in bt_lfw]}")
                        print(f"** TPR@FPR: {round(val, 6)}, FPR: {far}")

                        # 먼저, embeddings_eval에서 쌍별로 임베딩을 분리합니다. 
                        embeddings1 = adapted_embeddings_eval[0::2]
                        embeddings2 = adapted_embeddings_eval[1::2]
                        # 전체 쌍에 대해 거리를 계산
                        dists = distance(embeddings1, embeddings2, distance_metric=0)

                        # genuine 쌍(issame_list가 1인 경우)만 선택
                        issame_array = np.array(issame_list)
                        genuine_dists = dists[issame_array == 1]
                        imposter_dists = dists[issame_array == 0]

                        # genuine 쌍의 평균 거리를 계산
                        average_genuine_dist = np.mean(genuine_dists)
                        print("** Adpaeted Average distance for genuine pairs:", average_genuine_dist)
                        std_genuine_dists = np.std(genuine_dists)
                        print("** Adpaeted Standard deviation for genuine pairs:", std_genuine_dists)
                        average_imposter_dist = np.mean(imposter_dists)
                        print("** Adpaeted Average distance for imposter pairs:", average_imposter_dist)
                        std_imposter_dists = np.std(imposter_dists)
                        print("** Adpaeted Standard deviation for imposter pairs:", std_imposter_dists)

                        # Reset Current Model
                        del o_model, t_model, param
