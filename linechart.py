import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 中文支持
matplotlib.rcParams['font.family'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv("emotion_aggregate_results.csv")

# 表情编码：用于控制曲线形状（越靠近中性值越平稳）
emotion_to_code = {
    "Angry": -3,
    "Disgust": -2,
    "Fear": -1,
    "Sad": 0,
    "Neutral": 1,
    "Happy": 2,
    "Surprise": 3,
    "Unknown": None
}

# 中文标签映射（与上面一一对应）
code_to_chinese = {
    -3: "愤怒",
    -2: "厌恶",
    -1: "害怕",
     0: "悲伤",
     1: "中性",
     2: "开心",
     3: "惊讶"
}

# 构造完整时间序列
max_time = df['second'].max()
full_time = pd.DataFrame({'second': list(range(int(max_time) + 1))})
df_full = full_time.merge(df, on='second', how='left')

# 填补缺失值（前向填充）
for col in ['no_window', 'window', 'weighted']:
    df_full[col].fillna(method='ffill', inplace=True)
    df_full[col] = df_full[col].map(emotion_to_code)

# 每5秒采样
df_sampled = df_full[df_full['second'] % 5 == 0].reset_index(drop=True)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(df_sampled['second'], df_sampled['no_window'], label='无滑窗', linestyle='--', marker='o', color='red')
plt.plot(df_sampled['second'], df_sampled['window'], label='滑动窗口', linestyle='-.', marker='s', color='blue')
plt.plot(df_sampled['second'], df_sampled['weighted'], label='滑窗+加权', linestyle='-', marker='^', color='green')

# Y轴中文标签
yticks = sorted(code_to_chinese.keys())
plt.yticks(yticks, [code_to_chinese[i] for i in yticks])

plt.xlabel('时间（秒）')
plt.ylabel('主导情绪')
plt.title('不同策略下主导情绪随时间变化图（每5秒采样）')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
