import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Load the CSV data into a pandas DataFrame
file_path = "./10520_cart_knnr_model120_detailed_predictions_120frames.csv"  # ← 依實際路徑調整
df = pd.read_csv(file_path) 



numeric_cols = ['predicted_x', 'predicted_y', 'true_x', 'true_y', 'error_x', 'error_y', 'euclidean_error']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')



df = df.dropna(subset=['predicted_x', 'predicted_y', 'true_x', 'true_y'])

# Get unique sample_ids
sample_ids = df['sample_id'].unique()
print(sample_ids)

max_samples = 695
sample_ids = sample_ids[:min(max_samples, len(sample_ids))]


for sample_id in sample_ids:
    sample_df = df[df['sample_id'] == sample_id].sort_values('absolute_frame')

    # 取出時間序列（以絕對幀號或時間戳都行）
    t = sample_df['absolute_frame'].values
    
    # 做一個顏色映射：時間早 → 深色；時間晚 → 淺色
    norm = mcolors.Normalize(vmin=t.min(), vmax=t.max())
    cmap = cm.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(10, 6))

    # ① 預測軌跡（用散點 + 連線）
    ax.scatter(sample_df['predicted_x'], sample_df['predicted_y'],
               c=cmap(norm(t)), s=20, label='Predicted', zorder=2)
    ax.plot(sample_df['predicted_x'], sample_df['predicted_y'],
            linestyle='-', alpha=0.3, color='gray')

    # ② 真實軌跡
    ax.scatter(sample_df['true_x'], sample_df['true_y'],
               c=cmap(norm(t)), marker='x', s=30, label='True', zorder=2)
    ax.plot(sample_df['true_x'], sample_df['true_y'],
            linestyle='--', alpha=0.3, color='gray')

    # ③ 顏色條（time bar）
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Frame (time)')

    ax.set_title(f'Trajectory for Sample {sample_id}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    plt.show()
"""
# ===== 修改這區就好 =====
target_samples = [599, 587]          # 想跑 2 與 5 兩個 sample_id
# target_samples = [2]           # 想跑單一 sample_id = 2
# ==========================

for sample_id in target_samples:
    # 先確認這個 sample_id 是否真的存在
    if sample_id not in df['sample_id'].unique():
        print(f"Sample {sample_id} 不存在，跳過")
        continue

    object_ids = df[df['sample_id'] == sample_id]['object_id'].unique()

    for obj_id in object_ids:
        sample_df = (
            df[(df['sample_id'] == sample_id) & (df['object_id'] == obj_id)]
            .sort_values("absolute_frame")
        )

        plt.figure(figsize=(10, 6))
        plt.plot(sample_df['predicted_x'], sample_df['predicted_y'],
                 label='Predicted', marker='o', markersize=3)
        plt.plot(sample_df['true_x'], sample_df['true_y'],
                 label='True', marker='x', markersize=3)
        plt.title(f"Trajectory - Sample {sample_id} | Object {obj_id}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()
"""