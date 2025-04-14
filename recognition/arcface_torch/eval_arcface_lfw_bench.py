"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -*- coding: utf-8 -*-
import datetime
import os
import sys
import warnings

import numpy as np
import sklearn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


from backbones import get_model
from utils.utils_config import get_config


warnings.filterwarnings(("ignore"))


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


TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])


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


def get_embeddings_from_pathlist(model, path_list, batch_size=16):
    dataset = FaceDataset(path_list, transform=TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings_dict = {}
    with torch.no_grad():
        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            batch_embeds = model(batch_imgs)  # 배치 단위로 임베딩 계산
            batch_embeds = batch_embeds.cpu().numpy()  # CPU로 변환 후 numpy 배열로 변환

            for path, embed in zip(batch_paths, batch_embeds):
                embeddings_dict[path] = embed

    return embeddings_dict



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
"""
def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy
"""

"""
def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc
"""
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


"""
def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

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


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn, best_lfw_thres = calculate_roc(thresholds, 
                                                               embeddings1, 
                                                               embeddings2,
                                                               np.asarray(actual_issame), 
                                                               nrof_folds=nrof_folds,
                                                               distance_metric=distance_metric, 
                                                               subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far, best_ijb_thres = calculate_val(thresholds, 
                                                      embeddings1, 
                                                      embeddings2,
                                                      np.asarray(actual_issame), 
                                                      1e-3, nrof_folds=nrof_folds,
                                                      distance_metric=distance_metric, 
                                                      subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far, fp, fn, best_lfw_thres, best_ijb_thres
    # tpr, fpr, accuracy = calculate_roc(thresholds,
    #                                    embeddings1,
    #                                    embeddings2,
    #                                    np.asarray(actual_issame),
    #                                    nrof_folds=nrof_folds,
    #                                    pca=pca)
    # thresholds = np.arange(0, 4, 0.001)
    # val, val_std, far = calculate_val(thresholds,
    #                                   embeddings1,
    #                                   embeddings2,
    #                                   np.asarray(actual_issame),
    #                                   1e-3,
    #                                   nrof_folds=nrof_folds)
    # return tpr, fpr, accuracy, val, val_std, far


def distance(embeding1, embeding2, distance_metric=0):
    eps = 1e-6
    if distance_metric == 0:
        dot = np.sum(np.multiply(embeding1,embeding2), axis=1)# 벡터의 내적
        norm1 = np.linalg.norm(embeding1, ord=2, axis=1)
        norm2 = np.linalg.norm(embeding2, ord=2, axis=1)
        norm = norm1 * norm2 + eps  # 0 나누기를 피하기 위해 eps 추가
        cos_similarity = dot / norm

        dist = 1 - cos_similarity # 코사인 유사도 기반 거리 계산
        # dist = np.arccos(cos_similarity) / math.pi # 코사인 유사도 기반 각도를 0~1 사의로 정규화 = 아크코사인 사용
    elif distance_metric == 1:
        # 유클리드 거리 계산
        diff = np.subtract(embeding1, embeding2)
        dist = np.sqrt(np.sum(np.square(diff), axis=1))
        # 또는 아래 방법도 가능
        # dist = np.linalg.norm(embeding1 - embeding2, ord=2, axis=1)
    else:
        raise Exception("Undefined distance metirc %d" % distance_metric)
    return dist


def get_arcface_model(name='r50'):
    network = name
    if name == 'g_r50':
        model_config = 'configs/glint360k_r50.py'
        model_path = '../../model_zoo/glint360k_cos_r50.pth'
        network = 'r50'
    elif name == 'g_r100':
        model_config = 'configs/glint360k_r100.py'
        model_path = '../../model_zoo/glint360k_cos_r100.pth'
        network = 'r100'
    elif name == 'r50':
        model_config = 'configs/ms1mv3_r50.py'
        model_path = '../../model_zoo/ms1mv3_arc_r50.pth'
    elif name == 'r100':
        model_config = 'configs/ms1mv3_r100.py'
        model_path = '.,./../model_zoo/ms1mv3_arc_r100.pth'
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    cfg = get_config(model_config)
    model = get_model(network, dropout=0, fp16=True, num_features=cfg.embedding_size)
    weight = torch.load(model_path, weights_only=True)
    model.load_state_dict(weight)
    # model = torch.nn.DataParallel(model)
    print(f"** Model Weight loaded: {model_path}")

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data_dir', default='', help='')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--network', default='r50', type=str, help='')

    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    
    args = parser.parse_args()

    # 결과를 파일에 저장하기 위해 Tee 클래스 사용, 로그 파일명 설정 (원하는 경로로 수정 가능)
    log_file = os.path.join(os.path.dirname(__file__), './sample.txt')
    # 콘솔과 파일에 동시 출력하도록 설정
    sys.stdout = Tee(log_file)

    # 이미지 폴더 경로
    dataset_dir = args.data_dir
    print(f"** Test Dataset: {os.path.basename(dataset_dir)}")
    batch_size = args.batch_size

    device = torch.device(f'cuda:{args.gpu}'if torch.cuda.is_available() else 'cpu')
    print(f'** Running on device: {device}')

    model = get_arcface_model(args.network)
    model.eval()
    model.to(device)
    
    path_list, issame_list = get_paths(dataset_dir)
    embeddings_dict = get_embeddings_from_pathlist(model, path_list, batch_size=batch_size)
    embeddings_eval = np.array([embeddings_dict[path] for path in path_list])

    tpr, fpr, accuracy, val, val_std, far, fp, fn, bt_lfw, bt_ijb = evaluate(
        embeddings_eval, 
        issame_list, 
        nrof_folds=10,
        distance_metric=0,
        subtract_mean=False)
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
