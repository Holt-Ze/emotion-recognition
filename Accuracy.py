import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（适配 SimHei 黑体）
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['axes.unicode_minus'] = False

# 模拟训练轮次
epochs = np.arange(1, 51)

# 构造模拟准确率数据
train_acc = 0.6 + 0.4 * (1 - np.exp(-0.1 * epochs)) + np.random.normal(0, 0.005, size=50)
val_acc = 0.5 + 0.25 * (1 - np.exp(-0.08 * epochs)) + np.random.normal(0, 0.005, size=50)
test_acc = 0.5 + 0.22 * (1 - np.exp(-0.08 * epochs)) + np.random.normal(0, 0.005, size=50)

# 限制范围在0-1
train_acc = np.clip(train_acc, 0, 1)
val_acc = np.clip(val_acc, 0, 1)
test_acc = np.clip(test_acc, 0, 1)

# 绘图
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_acc * 100, label='训练集准确率', color='blue', linestyle='-', marker='o')
plt.plot(epochs, val_acc * 100, label='验证集准确率', color='green', linestyle='--', marker='s')
plt.plot(epochs, test_acc * 100, label='测试集准确率', color='red', linestyle='-.', marker='^')

# 标注验证集准确率基线
plt.axhline(y=72.4, color='gray', linestyle=':', linewidth=1, label='验证集稳定准确率：72.4%')

# 设置图表标题和标签
plt.title('表情识别模型训练过程中准确率变化')
plt.xlabel('训练轮数（Epoch）')
plt.ylabel('准确率（%）')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
