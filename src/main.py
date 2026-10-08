import cv2
import mediapipe as mp
import time

from pycaw.pycaw import AudioUtilities


# -----------------------------
# تنظیمات MediaPipe
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# -----------------------------
# کنترل صدای ویندوز
# -----------------------------

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

min_volume = volume.GetVolumeRange()[0]
max_volume = volume.GetVolumeRange()[1]


# -----------------------------
# دوربین
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened!")
    exit()


# زمان آخرین تغییر صدا
last_change = 0

# فاصله بین تغییرات صدا
change_delay = 0.8


# -----------------------------
# تابع شمارش انگشت‌ها
# -----------------------------

def count_fingers(hand_landmarks):

    fingers = 0

    landmarks = hand_landmarks.landmark

    # نوک انگشت‌ها
    tips = [8, 12, 16, 20]

    # مفصل‌های قبل از نوک انگشت
    pips = [6, 10, 14, 18]

    # چهار انگشت به جز شست
    for tip, pip in zip(tips, pips):

        if landmarks[tip].y < landmarks[pip].y:
            fingers += 1

    return fingers


# -----------------------------
# برنامه اصلی
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read camera!")
        break

    # برگرداندن تصویر مثل آینه
    frame = cv2.flip(frame, 1)

    # تبدیل BGR به RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # تشخیص دست
    result = hands.process(rgb_frame)

    finger_count = 0

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # رسم نقاط دست
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # شمارش انگشت‌ها
            finger_count = count_fingers(hand_landmarks)

    # -----------------------------
    # کنترل صدا
    # -----------------------------

    current_time = time.time()

    if current_time - last_change > change_delay:

        # 3 انگشت → زیاد کردن صدا
        if finger_count == 3:

            current_volume = volume.GetMasterVolumeLevel()

            new_volume = min(
                current_volume + 3,
                max_volume
            )

            volume.SetMasterVolumeLevel(new_volume, None)

            last_change = current_time

            print("Volume UP")

        # 2 انگشت → کم کردن صدا
        elif finger_count == 2:

            current_volume = volume.GetMasterVolumeLevel()

            new_volume = max(
                current_volume - 3,
                min_volume
            )

            volume.SetMasterVolumeLevel(new_volume, None)

            last_change = current_time

            print("Volume DOWN")

    # -----------------------------
    # نمایش تعداد انگشت‌ها
    # -----------------------------

    cv2.rectangle(
        frame,
        (20, 20),
        (300, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Fingers: {finger_count}",
        (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # راهنما
    cv2.putText(
        frame,
        "3 = Volume UP",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "2 = Volume DOWN",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # نمایش تصویر
    cv2.imshow("Hand Volume Control", frame)

    # خروج با Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# -----------------------------
# پایان برنامه
# -----------------------------

cap.release()
cv2.destroyAllWindows()