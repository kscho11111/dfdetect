import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
data = np.load('dataset.npz')
X_images_train = data['X_images_train']
X_images_test = data['X_images_test']
y_train = data['y_train']
y_test = data['y_test']

# PyTorch로 변환
X_images_train_tensor = torch.tensor(X_images_train, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
X_images_test_tensor = torch.tensor(X_images_test, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 데이터셋 및 데이터로더 생성
train_dataset = TensorDataset(X_images_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_images_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 정의
class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        self.base_model = models.efficientnet_b0(weights='DEFAULT')  # 최신 weights 사용
        num_features = self.base_model.classifier[-1].in_features  # 마지막 레이어의 입력 피쳐 수
        self.base_model.classifier = nn.Linear(num_features, 1)  # 1개의 출력 (binary classification)
        
    def forward(self, x):
        return self.base_model(x)

# 모델, 손실 함수 및 최적화 설정
model = EfficientNetModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 훈련
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # 검증
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
    
    val_loss = val_running_loss / len(test_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 평가
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).squeeze()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# 정확도와 손실 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

