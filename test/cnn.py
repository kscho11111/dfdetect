import cv2
import os
import shutil
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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
dataset_root = '/home/chokyungsoo/dfdetect/dataset'
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

# 데이터 크기 확인
print(f"X_images_real shape: {X_images_real.shape}")
print(f"X_images_fake shape: {X_images_fake.shape}")

# Flattening
X_images_real_flattened = X_images_real.reshape(-1, 64, 64, 3)
X_images_fake_flattened = X_images_fake.reshape(-1, 64, 64, 3)

y_real_flattened = np.zeros(X_images_real_flattened.shape[0])
y_fake_flattened = np.ones(X_images_fake_flattened.shape[0])

# X_images와 y 결합
X_images = np.concatenate((X_images_real_flattened, X_images_fake_flattened), axis=0)
y = np.concatenate((y_real_flattened, y_fake_flattened), axis=0)

# 데이터 크기 확인
print(f"Combined X_images shape: {X_images.shape}")
print(f"Combined y shape: {y.shape}")

# 데이터 분할
X_images_train, X_images_test, y_train, y_test = train_test_split(
    X_images, y, test_size=0.2, random_state=42
)

print("X_images_train shape:", X_images_train.shape)
print("X_images_test shape:", X_images_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 모델 구축
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  # 0 ya da 1 (gerçek ya da deepfake)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(
    X_images_train, y_train,
    epochs=10,  # 조정 가능
    batch_size=32,
    validation_data=(X_images_test, y_test)
)

# 평가
test_loss, test_acc = model.evaluate(X_images_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# 정확도와 손실 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

