import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# 1️⃣ 데이터 불러오기
print("📂 데이터 불러오는 중...")
X_train = np.load("X_train.npy") / 255.0  # 픽셀 값을 0~1로 정규화
y_train = np.load("y_train.npy", allow_pickle=True)  # 키 입력 데이터 불러오기

# 2️⃣ 키 입력을 원-핫 인코딩
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)

# 3️⃣ CNN 모델 생성
print("🧠 AI 모델 생성 중...")
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(150, 200, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(mlb.classes_), activation="sigmoid")  # 다중 키 입력 예측
])

# 4️⃣ 모델 컴파일 및 학습
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("🚀 AI 학습 시작...")
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 5️⃣ 모델 저장
model.save("ets2_ai_driver.h5")
print("✅ AI 학습 완료! 모델이 'ets2_ai_driver.h5'에 저장됨.")
