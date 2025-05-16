import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 设置随机种子（确保可重复性）
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 读取 CSV 文件
data_path = "fer2013.csv"
df = pd.read_csv(data_path)

# 数据清洗
df = df.dropna()
print("数据列名:", df.columns)

# 分割数据集
train_data = df[df['Usage'] == 'Training'].reset_index(drop=True)
val_data = df[df['Usage'] == 'PublicTest'].reset_index(drop=True)
test_data = df[df['Usage'] == 'PrivateTest'].reset_index(drop=True)


def parse_fer_data(df):
    images, labels = [], []
    for i in range(len(df)):
        pixels_str = df.loc[i, 'pixels']
        pixels = np.array(pixels_str.split(), dtype=np.uint8).reshape(48, 48)
        images.append(pixels)
        labels.append(df.loc[i, 'emotion'])
    return np.array(images), np.array(labels)


X_train, y_train = parse_fer_data(train_data)
X_val, y_val = parse_fer_data(val_data)
X_test, y_test = parse_fer_data(test_data)

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")


# 定义数据集和数据加载器
class FERDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = FERDataset(X_train, y_train, transform=train_transform)
val_dataset = FERDataset(X_val, y_val, transform=val_test_transform)
test_dataset = FERDataset(X_test, y_test, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)


# 模型定义
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)

# 输出模型结构
summary(model, (1, 48, 48))

# 训练参数
num_epochs = 20
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()  # 混合精度加速

best_val_acc = 0.0  # 跟踪最佳验证精度


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    global best_val_acc
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader, device)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc
            }, "emotion_model_best.pth")
            print(f"模型已保存在第 {epoch + 1} 个epoch，验证准确率: {val_acc:.2f}%")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# 训练并保存最佳模型
train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

# 最终测试
test_acc = evaluate(model, test_loader, device)
print(f"测试集准确率: {test_acc:.2f}%")

# 保存最终模型（可选）
torch.save(model.state_dict(), "emotion_model_final.pth")
