import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Residual Block 정의
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut  # Residual connection
        return nn.ReLU()(x)

# DCNN 모델 정의
class DCNNModel(nn.Module):
    def __init__(self):
        super(DCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dense Block 1
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Residual Block 1
        self.res_block1 = ResidualBlock(64)

        # Dense Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Residual Block 2
        self.res_block2 = ResidualBlock(128)

        # Dense Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        # Residual Block 3
        self.res_block3 = ResidualBlock(256)

        # Dense Block 4
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Residual Block 4
        self.res_block4 = ResidualBlock(512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)

        x = self.res_block1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.ReLU()(x)
        x = self.pool(x)

        x = self.res_block2(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.ReLU()(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.ReLU()(x)
        x = self.pool(x)

        x = self.res_block3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.ReLU()(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.ReLU()(x)
        x = self.pool(x)

        x = self.res_block4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 데이터 불러오기
data = np.load('dataset.npz')
X_images_train = data['X_images_train']
X_images_test = data['X_images_test']
y_train = data['y_train']
y_test = data['y_test']

# PyTorch로 변환
X_images_train_tensor = torch.tensor(X_images_train, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
X_images_test_tensor = torch.tensor(X_images_test, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # (N, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # (N, 1)

# 데이터셋 및 데이터로더 생성
train_dataset = TensorDataset(X_images_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_images_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델, 손실 함수 및 최적화 설정
model = DCNNModel()
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
    correct_train = 0
    total_train = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()  # (N, 1) -> (N,)
        loss = criterion(outputs, labels.view(-1))  # labels를 (N,)으로 변경
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 정확도 계산
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels.view(-1)).sum().item()  # labels를 (N,)으로 변경
        
        # 진행 상황 출력
        if (batch_idx + 1) % 10 == 0:  # 10번째 배치마다 출력
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}] - "
                  f"Loss: {loss.item():.4f}, Accuracy: {correct_train / total_train:.4f}")

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)

    # 검증
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.view(-1))  # labels를 (N,)으로 변경
            val_running_loss += loss.item()
            
            # 정확도 계산
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels.view(-1)).sum().item()  # labels를 (N,)으로 변경
    
    val_loss = val_running_loss / len(test_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} - "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
torch.save(model.state_dict(), 'dcnn_model.pth')
