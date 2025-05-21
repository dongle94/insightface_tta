import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20

# 데이터 로드
out_data = os.path.abspath("./npz/topofr_ms1mv2_r100_asian_celebrity.npz")
lfw_data = os.path.abspath("./npz/topofr_ms1mv2_r100_LFW.npz")
save_path = "./figure_topo-v2-r100.png"

data_our = np.load(out_data)
tp_our = data_our["tp"]
tn_our = data_our["tn"]
fp_our = data_our["fp"]
fn_our = data_our["fn"]
data_LFW = np.load(lfw_data)
tp_LFW = data_LFW["tp"]
tn_LFW = data_LFW["tn"]
fp_LFW = data_LFW["fp"]
fn_LFW = data_LFW["fn"]
eps = 1e-10
tpr_our = np.mean(tp_our / (tp_our + fn_our + eps), axis=0)
fpr_our = np.mean(fp_our / (fp_our + tn_our + eps), axis=0)

precision_our = np.mean(tp_our / (tp_our + fp_our + eps), axis=0)
recall_our = np.mean(tp_our / (tp_our + fn_our + eps), axis=0)

tpr_LFW = np.mean(tp_LFW / (tp_LFW + fn_LFW + eps), axis=0)
fpr_LFW = np.mean(fp_LFW / (fp_LFW + tn_LFW + eps), axis=0)

precision_LFW = np.mean(tp_LFW / (tp_LFW + fp_LFW + eps), axis=0)
recall_LFW = np.mean(tp_LFW / (tp_LFW + fn_LFW + eps), axis=0)

# 메인 PR Curve
plt.figure(figsize=(6, 5))

plt.plot(recall_our, precision_our, label='Ours', color='crimson', linewidth=2)
plt.plot(recall_LFW, precision_LFW, label='LFW', color='royalblue', linewidth=2)
plt.xlabel("Recall", fontsize=20, labelpad=10)
plt.ylabel("Precision", fontsize=20)
plt.rc('xtick', labelsize=16)  # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=18)  # y축 눈금 폰트 크기

plt.xlim(0.95, 1.0)
plt.ylim(0.5, 1.01)

plt.grid(True, alpha=0.5)
# plt.rc('font', size=16)        # 기본 폰트 크기
plt.legend(loc='lower left', fontsize=20)
plt.tight_layout()
plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
plt.show()