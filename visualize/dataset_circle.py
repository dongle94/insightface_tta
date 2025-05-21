import matplotlib.pyplot as plt

# 폰트 설정
plt.rcParams['font.family'] = 'Arial'  # 일반적으로 LaTeX 호환되는 폰트
plt.rcParams['font.size'] = 20  # 전체 폰트 크기 설정

# 데이터
labels = ['Female', 'Male']
sizes = [36, 34]
colors = ['crimson', 'royalblue']
explode = (0.05, 0)

# 그래프 생성
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    textprops={'fontsize': 20}
)

# 제목 설정
# plt.title('(b): Overall gender ratio', fontsize=18)
ax.axis('equal')  # 원형 유지

# PDF로 저장
output_path = "./figure_2b_gender.png"
plt.savefig(output_path, format='png', bbox_inches='tight')