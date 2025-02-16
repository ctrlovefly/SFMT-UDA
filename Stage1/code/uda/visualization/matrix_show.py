import matplotlib.pyplot as plt
import numpy as np

# 输入矩阵数据
matrix = np.array([[1, 0.92, 0.83, 0.67, 0.58, 0.5, 0.58, 0.75, 0.42, 0.75, 0.33, 0.25, 0.08, 0, 0.25, 0, 0],
    [0.92, 1, 0.92, 0.58, 0.67, 0.58, 0.67, 0.83, 0.42, 0.83, 0.42, 0.33, 0.25, 0.08, 0.33, 0.08, 0.08],
    [0.83, 0.92, 1, 0.5, 0.58, 0.67, 0.75, 0.92, 0.58, 0.92, 0.5, 0.42, 0.33, 0.17, 0.42, 0.17, 0.17],
    [0.67, 0.58, 0.5, 1, 0.92, 0.83, 0.75, 0.5, 0.75, 0.58, 0.5, 0.58, 0.5, 0.33, 0.08, 0.33, 0.33],
    [0.58, 0.67, 0.58, 0.92, 1, 0.92, 0.83, 0.67, 0.83, 0.67, 0.58, 0.67, 0.5, 0.42, 0.17, 0.42, 0.42],
    [0.5, 0.58, 0.67, 0.83, 0.92, 1, 0.92, 0.75, 0.92, 0.75, 0.67, 0.75, 0.67, 0.58, 0.25, 0.5, 0.5],
    [0.58, 0.67, 0.75, 0.75, 0.83, 0.92, 1, 0.67, 0.83, 0.67, 0.75, 0.67, 0.58, 0.42, 0.17, 0.42, 0.42],
    [0.75, 0.83, 0.92, 0.5, 0.67, 0.75, 0.67, 1, 0.75, 0.92, 0.42, 0.5, 0.42, 0.25, 0.5, 0.33, 0.33],
    [0.42, 0.42, 0.58, 0.75, 0.83, 0.92, 0.83, 0.75, 1, 0.67, 0.58, 0.67, 0.75, 0.58, 0.33, 0.58, 0.58],
    [0.75, 0.83, 0.92, 0.58, 0.67, 0.75, 0.67, 0.92, 0.67, 1, 0.42, 0.5, 0.42, 0.25, 0.5, 0.33, 0.33],
    [0.33, 0.42, 0.5, 0.5, 0.58, 0.67, 0.75, 0.2, 0.58, 0.42, 1, 0.92, 0.83, 0.67, 0.42, 0.67, 0.67],
    [0.25, 0.33, 0.42, 0.58, 0.67, 0.75, 0.67, 0.5, 0.67, 0.5, 0.92, 1, 0.92, 0.75, 0.5, 0.75, 0.75],
    [0.08, 0.25, 0.33, 0.5, 0.5, 0.67, 0.58, 0.42, 0.75, 0.42, 0.83, 0.92, 1, 0.83, 0.58, 0.83, 0.83],
    [0, 0.08, 0.17, 0.33, 0.42, 0.58, 0.42, 0.25, 0.58, 0.25, 0.67, 0.75, 0.83, 1, 0.75, 0.92, 0.92],
    [0.25, 0.33, 0.42, 0.08, 0.17, 0.25, 0.17, 0.5, 0.33, 0.5, 0.42, 0.5, 0.58, 0.75, 1, 0.75, 0.75],
    [0, 0.08, 0.17, 0.33, 0.42, 0.5, 0.42, 0.33, 0.58, 0.33, 0.67, 0.75, 0.83, 0.92, 0.75, 1, 0.92],
    [0, 0.08, 0.17, 0.33, 0.42, 0.5, 0.42, 0.33, 0.58, 0.33, 0.67, 0.75, 0.83, 0.92, 0.75, 0.92, 1]])

# LCZ标签
lcz_labels = ['LCZ1', 'LCZ2', 'LCZ3', 'LCZ4', 'LCZ5', 'LCZ6', 'LCZ7', 'LCZ8', 'LCZ9', 'LCZ10', 'LCZA', 'LCZB', 'LCZC', 'LCZD', 'LCZE', 'LCZF', 'LCZG']

# 创建绘图
fig, ax = plt.subplots(figsize=(13, 10))

# 绘制热图
# cmap = plt.cm.get_cmap('Blues')  # 使用 'Blues' 颜色映射
masked_matrix = np.ma.masked_where(np.abs(np.arange(matrix.shape[0])[:, None] - np.arange(matrix.shape[1])) > 3, matrix)
# heatmap = ax.imshow(masked_matrix, cmap=cmap, aspect='auto')
heatmap = ax.imshow(masked_matrix, cmap='summer', aspect='auto')

# 添加颜色条（bar 图例）
cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label('Value', rotation=270, labelpad=15, fontsize=15)  # 设置图例标签
cbar.ax.tick_params(labelsize=15)

# 在每个格子上添加对应的数字
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if abs(i - j) <= 3:  # 只显示对角线周围3个范围的数字
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=15)

# 添加网格线（可选）
ax.grid(visible=False)  # 取消网格线显示

# 添加刻度和标签
ax.set_xticks(np.arange(matrix.shape[1]))
ax.set_yticks(np.arange(matrix.shape[0]))
ax.set_xticklabels(lcz_labels, rotation=45, ha='right', fontsize=15)
ax.set_yticklabels(lcz_labels, fontsize=15)

# 添加标题
# ax.set_title('Similarity metric for LCZ classes')

# 调整布局
plt.tight_layout()
# plt.show()
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
