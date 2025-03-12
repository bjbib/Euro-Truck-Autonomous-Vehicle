import mss
import numpy as np
import cv2

with mss.mss() as sct:
    monitor = sct.monitors[1]  # 첫 번째 모니터 전체 캡처

    while True:
        screenshot = sct.grab(monitor)  # 화면 캡처
        img = np.array(screenshot)  # NumPy 배열로 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 색상 변환

        cv2.imshow("Screen Capture", img)  # 화면 표시

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
            break

cv2.destroyAllWindows()