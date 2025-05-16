import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from collections import deque
import math

# 表情标签
emotion_labels = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

# 数值编码映射，方便图表绘制
emotion_encoding = {
    "Angry": -3, "Disgust": -2, "Fear": -1, "Neutral": 0,
    "Happy": 1, "Sad": 2, "Surprise": 3
}

# 加载模型
class EmotionCNN(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(128 * 6 * 6, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# 加载权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotion_model_best.pth", map_location=device)["model_state_dict"])
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 视频
video_path = "her片段.mov"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

# Haar人脸
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 结果记录
results = []

# 滑动窗口
window = deque(maxlen=30)

def compute_dominant_emotion(window):
    count = {}
    for item in window:
        emo = item["emotion"]
        count[emo] = count.get(emo, 0) + 1
    if count:
        return max(count.items(), key=lambda x: x[1])[0]
    return "Neutral"

def compute_weighted_dominant_emotion(window, alpha=2.0, lambd=0.1):
    score = {}
    t_now = window[-1]["second"] if window else 0
    for item in window:
        emo = item["emotion"]
        c = item["confidence"]
        t_i = item["second"]
        w_conf = math.exp(alpha * (c - 0.5))
        w_time = math.exp(-lambd * (t_now - t_i))
        w_total = w_conf * w_time
        score[emo] = score.get(emo, 0) + w_total
    if score:
        return max(score.items(), key=lambda x: x[1])[0]
    return "Neutral"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % int(fps) == 0:  # 每秒1帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y + h, x:x + w]
            face_pil = Image.fromarray(face_roi)
            input_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            emotion = emotion_labels[predicted.item()]
            confidence_val = float(confidence.item())
            sec = int(frame_count // fps)

            # 添加入窗口
            window.append({
                "second": sec,
                "emotion": emotion,
                "confidence": confidence_val
            })

            # 计算三种策略结果
            result = {
                "second": sec,
                "no_window": emotion,
                "window": compute_dominant_emotion(window),
                "weighted": compute_weighted_dominant_emotion(window)
            }
            results.append(result)

    frame_count += 1

cap.release()

# 保存CSV
df = pd.DataFrame(results)
df.to_csv("emotion_aggregate_results.csv", index=False)
print("识别完成，结果已保存为 emotion_aggregate_results.csv")
