import streamlit as st
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from dcnn import DCNNModel  # dcnn.py와 같은 디렉토리에 있어야 합니다.

# 모델 로드
model = DCNNModel()
model.load_state_dict(torch.load('dcnn_model.pth'))
model.eval()

# 이미지 전처리 함수
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),  # 모델 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # 배치 차원 추가

# Streamlit 애플리케이션
st.title("Real or Fake Image Classifier")
st.write("사진을 업로드하여 실제(real)인지 가짜(fake)인지 확인하세요.")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드한 이미지.', use_column_width=True)
    
    # 이미지 전처리
    input_tensor = preprocess_image(image)
    
    # 예측 수행
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
    
    # 결과 출력
    if prediction > 0.5:
        st.write("결과: 가짜(fake)입니다.")
    else:
        st.write("결과: 실제(real)입니다.")

