# 基于深度学习的表情识别系统

## 项目简介
本项目是一个基于深度学习的实时表情识别系统，能够通过摄像头实时捕捉人脸并识别其表情状态。系统采用CNN深度学习模型，可以识别7种基本表情：愤怒、厌恶、恐惧、开心、悲伤、惊讶和中性。

## 功能特点
- 实时人脸检测和表情识别
- 支持多种表情分析策略：
  - 无滑窗：直接输出每帧的识别结果
  - 滑动窗口：通过时间窗口平滑识别结果
  - 加权分析：结合时间窗口和权重进行更稳定的预测
- 可视化分析工具：支持表情识别结果的图表展示

## 项目结构
```
├── emotion.py          # 核心模型和识别逻辑
├── video.py            # 视频处理和实时识别
├── train.py            # 模型训练脚本
├── Accuracy.py         # 准确率评估
├── linechart.py        # 结果可视化
├── environment.yml     # 环境配置文件
└── haarcascade_frontalface_default.xml  # 人脸检测模型
```

## 环境配置
1. 创建并激活conda环境：
```bash
conda env create -f environment.yml
conda activate emotion
```

2. 主要依赖：
- Python 3.8+
- PyTorch
- OpenCV
- Pandas
- Matplotlib
- NumPy

## 使用说明
1. 实时表情识别：
```bash
python video.py
```

2. 训练模型：
```bash
python train.py
```

3. 查看识别结果分析：
```bash
python linechart.py
```

## 模型说明
项目使用了自定义的CNN网络结构，包含：
- 3个卷积层，每层后接ReLU激活和最大池化
- 2个全连接层，带有Dropout正则化
- 输出层为7分类（对应7种基本表情）

## 注意事项
- 模型文件较大，未包含在代码仓库中
- 使用前请确保摄像头正常工作
- 建议在光线充足的环境下使用

## 许可证
MIT License