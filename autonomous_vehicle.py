import cv2
import numpy as np
import mss
import time
import keyboard
import threading
import pandas as pd

# 1️⃣ 화면 녹화 설정 (2560x1440 해상도)
screen_width = 2560
screen_height = 1440
fps = 20  # 초당 프레임 수
output_filename = "record.avi"

# 2️⃣ 동영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_filename, fourcc, fps, (screen_width, screen_height))

recording = True  # 녹화 상태 플래그


# 3️⃣ 화면 녹화 함수
def record_screen():
    global recording
    with mss.mss() as sct:
        monitor = {"top": 100, "left": 100, "width": screen_width, "height": screen_height}
        print("🎥 녹화 시작... (Ctrl + Q를 눌러 종료)")

        while recording:
            start_time = time.time()
            screen = sct.grab(monitor)
            img = np.array(screen)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            out.write(img)

            elapsed_time = time.time() - start_time
            sleep_time = max(1 / fps - elapsed_time, 0)
            time.sleep(sleep_time)


# 4️⃣ 키 입력 감지 함수
def log_keys():
    global recording
    with open("keys.txt", "a") as f:
        while recording:
            keys = []
            if keyboard.is_pressed("w"):
                keys.append("w")
            if keyboard.is_pressed("a"):
                keys.append("a")
            if keyboard.is_pressed("s"):
                keys.append("s")
            if keyboard.is_pressed("d"):
                keys.append("d")

            if keys:
                f.write(f"{time.time()} {keys}\n")

            time.sleep(0.1)


# 5️⃣ 멀티스레드 실행 (녹화 + 키 입력 감지)
thread1 = threading.Thread(target=record_screen)
thread2 = threading.Thread(target=log_keys)

thread1.start()
thread2.start()

# 6️⃣ Ctrl + Q 입력 대기
keyboard.wait("ctrl+q")

# 7️⃣ 녹화 및 키 입력 감지 종료
recording = False
thread1.join()
thread2.join()
out.release()
cv2.destroyAllWindows()
print("\n🚀 녹화 종료됨! 데이터 전처리 시작...")


# 8️⃣ 전처리 (record.avi + keys.txt → X_train.npy, y_train.npy)
def preprocess_data():
    print("🔄 데이터 전처리 중...")
    cap = cv2.VideoCapture("record.avi")
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    try:
        # 키 입력 데이터 불러오기
        keys_data = pd.read_csv("keys.txt", sep=" ", header=None, names=["timestamp", "keys"])

        # 만약 키 입력 데이터가 비어 있다면 예외 처리
        if keys_data.empty:
            print("⚠️ 경고: 키 입력 데이터가 비어 있습니다! 다시 녹화해주세요.")
            return

        frame_list = []
        label_list = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            key_row = keys_data.iloc[(keys_data["timestamp"] - timestamp).abs().idxmin()]

            frame_list.append(cv2.resize(frame, (200, 150)))  # 이미지 크기 축소
            label_list.append(key_row["keys"])

            frame_idx += 1

        cap.release()

        np.save("X_train.npy", np.array(frame_list))
        np.save("y_train.npy", np.array(label_list))

        print("✅ 전처리 완료! (X_train.npy, y_train.npy 생성)")

    except Exception as e:
        print(f"❌ 데이터 전처리 중 오류 발생: {e}")


# 9️⃣ 전처리 실행
preprocess_data()
