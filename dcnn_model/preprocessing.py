import cv2
import os
import shutil
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# 비디오에서 프레임 추출하는 함수
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

# 데이터 증강 함수
def augment_images(images):
    augmented_images = []
    for image in images:
        # 원본 이미지 추가
        augmented_images.append(image)
        
        # 회전
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(rotated)

        # 수평 반전
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)

        # 밝기 조절
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # 밝기 증가
        augmented_images.append(bright)

    return np.array(augmented_images)

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
for filename in file_list_real[:500]:
    source_file = os.path.join(source_dir_real, filename)
    destination_file = os.path.join(destination_dir_real, filename)
    shutil.copy(source_file, destination_file)

file_list_fake = os.listdir(source_dir_fake)
for filename in file_list_fake[:500]:
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
X_images_real_flattened = X_images_real.reshape(-1, 64, 64, 3)
X_images_fake_flattened = X_images_fake.reshape(-1, 64, 64, 3)

# 데이터 증강
X_images_real_augmented = augment_images(X_images_real_flattened)
X_images_fake_augmented = augment_images(X_images_fake_flattened)

# 라벨 생성
y_real_flattened = np.zeros(X_images_real_flattened.shape[0] + X_images_real_augmented.shape[0])
y_fake_flattened = np.ones(X_images_fake_flattened.shape[0] + X_images_fake_augmented.shape[0])

# X_images와 y 결합
X_images = np.concatenate((X_images_real_flattened, X_images_real_augmented), axis=0)
X_images = np.concatenate((X_images, X_images_fake_flattened), axis=0)
X_images = np.concatenate((X_images, X_images_fake_augmented), axis=0)

y = np.concatenate((y_real_flattened, y_fake_flattened), axis=0)

# 데이터 크기 확인
print(f"Combined X_images shape: {X_images.shape}")
print(f"Combined y shape: {y.shape}")

# 데이터 분할
X_images_train, X_images_test, y_train, y_test = train_test_split(
    X_images, y, test_size=0.2, random_state=42
)

# NumPy 파일로 저장
np.savez('dataset.npz', X_images_train=X_images_train, X_images_test=X_images_test, y_train=y_train, y_test=y_test)
print("Data saved successfully.")

