import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 비디오에서 프레임 추출 함수
def extract_video_frames(video_path, image_size, num_frames=10):
    video_capture = cv2.VideoCapture(video_path)
    frames = []

    for _ in range(num_frames):
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size)
        frames.append(frame)

    video_capture.release()
    return np.array(frames)

# 데이터셋 클래스 정의
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, image_size=(64, 64), num_frames=10):
        self.video_paths = video_paths
        self.labels = labels
        self.image_size = image_size
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = extract_video_frames(video_path, self.image_size, self.num_frames)
        # 프레임을 (C, H, W) 형태로 변환
        frames = frames.transpose((3, 0, 1, 2))  # (num_frames, H, W, C) -> (C, num_frames, H, W)
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Residual Block 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += shortcut  # Residual connection
        return F.relu(out)

# DCNN 모델 정의
class DeepFakeDetectorDCNN(nn.Module):
    def __init__(self):
        super(DeepFakeDetectorDCNN, self).__init__()
        self.input_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dense Block 1
        self.dense_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Residual Block 1
        self.residual_block1 = ResidualBlock(64, 64)

        # Dense Block 2
        self.dense_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Residual Block 2
        self.residual_block2 = ResidualBlock(128, 128)

        # Dense Block 3
        self.dense_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Residual Block 3
        self.residual_block3 = ResidualBlock(256, 256)

        # Dense Block 4
        self.dense_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Residual Block 4
        self.residual_block4 = ResidualBlock(512, 512)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.input_layer(x)))
        x = self.maxpool(x)

        # Dense Block 1
        x = self.dense_block1(x)
        x = self.residual_block1(x)

        # Dense Block 2
        x = self.dense_block2(x)
        x = self.residual_block2(x)

        # Dense Block 3
        x = self.dense_block3(x)
        x = self.residual_block3(x)

        # Dense Block 4
        x = self.dense_block4(x)
        x = self.residual_block4(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x)

# 손실 및 정확도 시각화 함수
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, epochs):
    plt.figure(figsize=(12, 6))

    # 손실 시각화
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'bo', label='Training loss')
    plt.plot(range(1, epochs + 1), val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 정확도 시각화
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, 'bo', label='Training accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_curve_dcnn.png')
    plt.show()
