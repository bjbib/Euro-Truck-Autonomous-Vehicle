import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import time
import mss
from sklearn.preprocessing import MultiLabelBinarizer

# 1️⃣ 학습된 모델 불러오기
print("📂 AI 모델 불러오는 중...")
model = tf.keras.models.load_model("ets2_ai_driver.h5")

# 2️⃣ 원-핫 인코딩 복원 (키 매핑)
mlb = MultiLabelBinarizer()
mlb.fit([['w'], ['a'], ['s'], ['d'], ['w', 'a'], ['w', 'd'], ['a', 's'], ['d', 's']])

# 3️⃣ 화면 캡처 함수 (유로트럭 게임 화면 가져오기)
def capture_screen():
    with mss.mss() as sct:
        monitor = {"top": 100, "left": 100, "width": 2560, "height": 1440}  # 해상도 맞추기
        screen = sct.grab(monitor)
        img = np.array(screen)
        img = cv2.resize(img, (200, 150))  # 학습된 모델 크기에 맞게 조정
        img = img / 255.0  # 정규화
        return img

# 4️⃣ AI 운전 실행
print("🚛 AI 운전 시작... (Ctrl + Q를 누르면 종료)")
running = True

def drive():
    global running
    while running:
        # 5️⃣ 화면 캡처 & AI 예측
        screen = capture_screen()
        screen = np.expand_dims(screen, axis=0)  # 배치 차원 추가
        prediction = model.predict(screen)[0]

        # 6️⃣ 예측된 키 입력 변환
        predicted_keys = mlb.inverse_transform((prediction > 0.5).astype(int))
        keys_to_press = predicted_keys[0] if predicted_keys else []

        # 7️⃣ 키 입력 수행
        for key in ["w", "a", "s", "d"]:
            if key in keys_to_press:
                pyautogui.keyDown(key)
            else:
                pyautogui.keyUp(key)

        time.sleep(0.1)  # 0.1초마다 입력 갱신

# 8️⃣ Ctrl + Q를 누르면 종료
import keyboard
keyboard.add_hotkey("ctrl+q", lambda: stop())

def stop():
    global running
    running = False
    print("\n🛑 AI 운전 종료!")

# 9️⃣ 실행
drive()
