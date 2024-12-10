import cv2
import os
import shutil
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 활성화 함수 사용을 위한 import

# 비디오 프레임 추출 함수
def extract_video_frames(video_path, image_size, num_frames=10):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    
    for frame_num in range(min(frame_count, num_frames)):
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size)
        frames.append(frame)

    video_capture.release()  # 비디오 캡처 객체 해제
    return frames

# 데이터셋 경로 설정
source_dir_real = '/home/chokyungsoo/dfdetect/celeb-df-v2/Celeb-real'
source_dir_fake = '/home/chokyungsoo/dfdetect/celeb-df-v2/Celeb-synthesis'
destination_dir_real = '/home/chokyungsoo/dfdetect/dataset/Celeb-real'
destination_dir_fake = '/home/chokyungsoo/dfdetect/dataset/Celeb-synthesis'

# 디렉토리 생성
os.makedirs(destination_dir_real, exist_ok=True)
os.makedirs(destination_dir_fake, exist_ok=True)

# 파일 복사
file_list_real = os.listdir(source_dir_real)
for filename in file_list_real[:100]:
    source_file = os.path.join(source_dir_real, filename)
    destination_file = os.path.join(destination_dir_real, filename)
    shutil.copy(source_file, destination_file)

file_list_fake = os.listdir(source_dir_fake)
for filename in file_list_fake[:100]:
    source_file = os.path.join(source_dir_fake, filename)
    destination_file = os.path.join(destination_dir_fake, filename)
    shutil.copy(source_file, destination_file)

print("Files copied successfully.")

# 비디오 경로 설정
real_video_paths = glob.glob(os.path.join(destination_dir_real, "*.mp4"))
fake_video_paths = glob.glob(os.path.join(destination_dir_fake, "*.mp4"))

# 프레임 추출
image_size = (64, 64)
num_samples = 100

X_images_real = []
X_images_fake = []

# 실제 비디오에서 프레임 추출
for video_path in real_video_paths[:num_samples]:
    frames = extract_video_frames(video_path, image_size)
    print(f"Extracted {len(frames)} frames from {video_path}")
    if len(frames) > 0:
        X_images_real.append(frames)

# 가짜 비디오에서 프레임 추출
for video_path in fake_video_paths[:num_samples]:
    frames = extract_video_frames(video_path, image_size)
    print(f"Extracted {len(frames)} frames from {video_path}")
    if len(frames) > 0:
        X_images_fake.append(frames)

# NumPy 배열로 변환
X_images_real = np.array(X_images_real)
X_images_fake = np.array(X_images_fake)

# Flattening
X_images_real_flattened = X_images_real.reshape(-1, 3, 64, 64)  # 채널 순서 (C, H, W)
X_images_fake_flattened = X_images_fake.reshape(-1, 3, 64, 64)

y_real_flattened = np.zeros(X_images_real_flattened.shape[0])
y_fake_flattened = np.ones(X_images_fake_flattened.shape[0])

# X_images와 y 결합
X_images = np.concatenate((X_images_real_flattened, X_images_fake_flattened), axis=0)
y = np.concatenate((y_real_flattened, y_fake_flattened), axis=0)

# 데이터셋 클래스 정의
class VideoDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 데이터셋 및 데이터로더 생성
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = VideoDataset(X_images, y, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 모델 훈련
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.float())
        labels = labels.view(-1, 1).float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 평가
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images.float())
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.view(-1) == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 정확도와 손실 시각화는 필요시 추가 가능합니다.

