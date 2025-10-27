import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

# =================== 配置参数 ===================
CSV_PATH = '/zs_test_5w_labels.csv'
NPZ_FOLDER = '/test_feature/'
OUTPUT_FOLDER = './tsne_visualizations_all'
FEATURE_KEY_IN_NPZ = 'arr_0'

N_JOBS = 16  # 根据 CPU 核心数调整

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =================== 中文字体修复：防止“口”字出现 ===================
# 方法：指定支持中文的字体，并关闭负号替换
plt.rcParams['font.sans-serif'] = [
    'SimHei',          # 常见黑体
    'Arial Unicode MS', # macOS 支持
    'DejaVu Sans',     # Matplotlib 内置
    'WenQuanYi Micro Hei', # Linux 常见
    'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 检查可用字体（可选调试）
# import matplotlib
# print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))

# =================== 1. 读取 CSV ===================
print("Loading CSV file...")
df = pd.read_csv(CSV_PATH, sep=',')
print(f"Loaded {len(df)} samples.")

disease_columns = df.columns[3:].tolist()
print(f"Diseases ({len(disease_columns)}): {disease_columns}")

# =================== 2. 加载特征向量 ===================
print("Loading features from .npz files...")
features = []
labels_dict = {disease: [] for disease in disease_columns}

for _, row in df.iterrows():
    img_name = row['image_name']
    npz_path = os.path.join(NPZ_FOLDER, f"{img_name}.npz")

    if not os.path.exists(npz_path):
        continue  # 跳过缺失文件

    try:
        data = np.load(npz_path)
        feat = data[FEATURE_KEY_IN_NPZ]
        if feat.ndim > 1:
            feat = feat.flatten()
        features.append(feat)

        for disease in disease_columns:
            labels_dict[disease].append(int(row[disease]))

    except Exception as e:
        print(f"Error loading {npz_path}: {e}")

if len(features) == 0:
    raise ValueError("No features loaded.")

features = np.array(features)
# print(f"Feature matrix shape: {features.shape}")


print(f"Original feature shape: {features.shape}")  # e.g., (50000, 512)

# ------------------- 配置采样参数 -------------------
MAX_SAMPLES = 5000000  # 最大保留样本数，建议 3000~10000
STRATIFIED = False   # 是否按标签分布分层采样（推荐 True）

if len(features) <= MAX_SAMPLES:
    print("No sampling needed: sample count <= MAX_SAMPLES")
    indices = np.arange(len(features))
else:
    if STRATIFIED:
        print(f"Performing stratified sampling (N={MAX_SAMPLES})...")
        from sklearn.model_selection import train_test_split

        # 使用任意一个疾病作为标签（这里用第一个）
        first_disease = disease_columns[0]
        y_stratify = np.array(labels_dict[first_disease])

        # 利用 train_test_split 按比例分层采样
        _, _, idx_train, idx_test = train_test_split(
            features, np.arange(len(features)),
            train_size=MAX_SAMPLES,
            stratify=y_stratify,
            random_state=42,
            shuffle=True
        )
        indices = idx_test
    else:
        print(f"Performing random sampling (N={MAX_SAMPLES})...")
        indices = np.random.choice(len(features), size=MAX_SAMPLES, replace=False)
        indices = np.sort(indices)  # 可选：保持顺序一致

print(f"Sampled {len(indices)} indices.")

# ------------------- 应用采样到 features 和所有 labels -------------------
features = features[indices]
print(f"Final feature shape after sampling: {features.shape}")

# 对每个疾病的标签也做相同索引筛选
for disease in disease_columns:
    labels_dict[disease] = [labels_dict[disease][i] for i in indices]

print("Feature and label sampling completed.")
# =================== 3. 单任务函数：t-SNE + 小散点图 ===================
def run_tsne_for_disease(args):
    disease, y, features, output_folder = args
    y = np.array(y)
    
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return f"Skipped: {disease} → only one class ({unique_labels[0]})"

    try:
        # 【可选】如果样本太多，先 PCA 降维加速
        if features.shape[1] > 500000:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features

        # t-SNE 降维
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(features) - 1),
            max_iter=1000,
            random_state=42,
            init='pca',
            learning_rate='auto'
        )
        emb = tsne.fit_transform(features_reduced)

                # 绘图：仅显示散点，无任何文字或边框
        plt.figure(figsize=(10, 8), dpi=300)
        
        # 分别绘制阴性和阳性点
        plt.scatter(
            emb[y == 0, 0], emb[y == 0, 1],
            c='#6CA6CD',
            s=1.0,
            alpha=0.6,
            rasterized=True
        )
        plt.scatter(
            emb[y == 1, 0], emb[y == 1, 1],
            c='#FF6347',
            s=1.0,
            alpha=0.6,
            rasterized=True
        )

        # 移除所有坐标轴元素
        plt.axis('off')
        # 调整子图参数以填满整个图像区域
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 保存为高分辨率图像，透明背景可选（默认不透明）
        safe_disease = "".join(c for c in disease if c.isalnum() or c in "_- ")
        output_path = os.path.join(output_folder, f"tsne_{safe_disease}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, format='png')
        plt.close()

        return f"Done: {disease} → saved to {output_path}"

    except Exception as e:
        return f"Failed: {disease} → {str(e)}"

# =================== 4. 并行执行所有任务 ===================
print(f"Starting parallel t-SNE for {len(disease_columns)} diseases using {N_JOBS} processes...")

tasks = [(disease, labels_dict[disease], features, OUTPUT_FOLDER) for disease in disease_columns]

results = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(run_tsne_for_disease)(task) for task in tasks
)

# 输出结果
print("\n" + "="*60)
for r in results:
    print(r)
