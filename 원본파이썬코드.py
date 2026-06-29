import cv2
import numpy as np
import serial
import time

# ====== 사용자 설정 ======
CAM_INDEX = 1                # Iriun 카메라 인덱스(0/1/2 바꿔서 맞추기)
COM_PORT = "COM25"             # 아두이노 포트로 수정
BAUD = 9600

WIDTH, HEIGHT = 640, 480
CX0, CY0 = WIDTH // 2, HEIGHT // 2

# ====== 로봇팔 각도/제한(너 로봇팔에 맞게 조절 권장) ======
base, arm1, arm2, grip = 90, 120, 90, 10

BASE_MIN, BASE_MAX = 0, 180
ARM1_MIN, ARM1_MAX = 30, 140
ARM2_MIN, ARM2_MAX = 50, 130
GRIP_MIN, GRIP_MAX = 0, 110

# ====== HSV 임계값(환경마다 튜닝 필요) ======
HSV = {
    "BLUE":  (np.array([95,  80,  80]), np.array([130, 255, 255])),
    "GREEN": (np.array([40,  80,  80]), np.array([85,  255, 255])),
    "RED1":  (np.array([0,  120, 80]),  np.array([10,  255, 255])),
    "RED2":  (np.array([170,120,80]),   np.array([180, 255, 255])),
}
kernel = np.ones((5, 5), np.uint8)

# [중요] 루프 밖(상단)에 아래 변수들을 먼저 선언해주세요
history_x = []
history_y = []
STABLE_COUNT = 20  # 몇 프레임 동안 데이터를 모을 것인가

# ====== 상태 머신 ======
SEARCH, TRACK, GRAB, PLACE = "SEARCH", "TRACK", "GRAB", "PLACE"
state = SEARCH

target_color = "BLUE"
sweep_dir = +1
last_sweep = time.time()

# ====== 판정/튜닝 파라미터 ======
AREA_MIN = 900
AREA_CLOSE = 14000
CENTER_TOL_X = 35
CENTER_TOL_Y = 35

# 추적 민감도(너무 흔들리면 KX/KY를 더 작게)
KX = 1 / 70
KY = 1 / 90

# 한 프레임에서 바뀌는 최대 각도 제한(진동 줄이기)
MAX_STEP_BASE = 4
MAX_STEP_ARM1 = 3

GRAB_TIME = 1.0
PLACE_TIME = 1.2

grab_t0 = None
place_t0 = None

HOME_PRESET = (90, 120, 90, 10)
PLACE_PRESET = {
    "RED":   (50,  120, 90, 10),
    "GREEN": (90,  120, 90, 10),
    "BLUE":  (130, 120, 90, 10),
}

# ====== 전송 주기 제한(너무 자주 보내면 서보 떨림/포트부하) ======
SEND_PERIOD = 0.03   # 30ms = 약 33Hz
_last_send = 0.0
_last_sent_angles = None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def step_limit(delta, max_step):
    if delta > max_step:
        return max_step
    if delta < -max_step:
        return -max_step
    return delta

def make_mask(frame_bgr, target):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if target == "RED":
        m1 = cv2.inRange(hsv, HSV["RED1"][0], HSV["RED1"][1])
        m2 = cv2.inRange(hsv, HSV["RED2"][0], HSV["RED2"][1])
        mask = m1 | m2
    else:
        lo, hi = HSV[target]
        mask = cv2.inRange(hsv, lo, hi)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def detect_candidates(frame_bgr, target):
    mask = make_mask(frame_bgr, target)
    mask[0:int(HEIGHT*0.4), :] = 0
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < AREA_MIN:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        cands.append((area, cx, cy, x, y, w, h))
    return cands, mask

def select_best(cands, rule="largest"):
    if not cands:
        return None
    if rule == "largest":
        return max(cands, key=lambda t: t[0])
    return max(cands, key=lambda t: -abs(t[1]-CX0) - abs(t[2]-CY0))

def centered(cx, cy):
    return abs(cx - CX0) <= CENTER_TOL_X and abs(cy - CY0) <= CENTER_TOL_Y

def close_enough(area):
    return area >= AREA_CLOSE

def _raw_send(ser, b, a1, a2, g):
    msg = f"{b},{a1},{a2},{g}\n"
    ser.write(msg.encode("utf-8"))

def send_cmd_throttled(ser, b, a1, a2, g, force=False):
    """전송 주기 제한 + 같은 값 반복 전송 방지"""
    global _last_send, _last_sent_angles
    now = time.time()
    angles = (int(b), int(a1), int(a2), int(g))

    if not force:
        if (now - _last_send) < SEND_PERIOD:
            return
        if _last_sent_angles == angles:
            _last_send = now
            return

    _raw_send(ser, *angles)
    _last_send = now
    _last_sent_angles = angles

def sweep_base(b):
    global sweep_dir, last_sweep
    now = time.time()
    if now - last_sweep < 0.05:
        return b
    last_sweep = now
    b += sweep_dir * 2
    if b >= BASE_MAX:
        b = BASE_MAX
        sweep_dir = -1
    elif b <= BASE_MIN:
        b = BASE_MIN
        sweep_dir = +1
    return b

def go_home(ser):
    """종료/에러 시 안전 복귀"""
    hb, ha1, ha2, hg = HOME_PRESET
    send_cmd_throttled(ser, hb, ha1, ha2, hg, force=True)

# ====== 메인 ======
ser = None
cap = None

try:
    # 시리얼 오픈(오픈 시 보드가 리셋될 수 있음)
    ser = serial.Serial(COM_PORT, BAUD, timeout=0.1)
    time.sleep(2.0)

    # 시작하자마자 HOME 한 번 보내서 상태 맞추기
    go_home(ser)
    time.sleep(0.2)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("카메라 열기 실패: CAM_INDEX를 0/1/2로 바꿔보세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            target_color = "RED"
        elif key == ord('2'):
            target_color = "GREEN"
        elif key == ord('3'):
            target_color = "BLUE"
        elif key == ord('q'):
            break

        cands, mask = detect_candidates(frame, target_color)
        best = select_best(cands, rule="largest")

        seen = best is not None
        if seen:
            area, cx, cy, x, y, w, h = best
            is_center = centered(cx, cy)
            is_close = close_enough(area)
        else:
            area, cx, cy = 0, None, None
            is_center, is_close = False, False

        # ====== 상태 전이 ======
        if state == SEARCH:
            if seen:
                state = TRACK
        elif state == TRACK:
            if not seen:
                state = SEARCH
            elif is_center and is_close:
                state = GRAB
                grab_t0 = time.time()
        elif state == GRAB:
            if grab_t0 is not None and (time.time() - grab_t0) >= GRAB_TIME:
                state = PLACE
                place_t0 = time.time()
        elif state == PLACE:
            if place_t0 is not None and (time.time() - place_t0) >= PLACE_TIME:
                state = SEARCH
                grab_t0 = None
                place_t0 = None

        # ====== 상태별 제어 ======
        if state == SEARCH:
            base = sweep_base(base)
            arm1, arm2, grip = HOME_PRESET[1], HOME_PRESET[2], HOME_PRESET[3]

        elif state == TRACK:
            # 안전: seen이 False면 cx/cy가 None이므로 TRACK에서만 계산
            err_x = (cx - CX0)
            err_y = (cy - CY0)

            d_base = int(err_x * KX * 100)
            d_arm1 = int(-err_y * KY * 100)

            # 한 번에 너무 많이 움직이지 않게 제한
            d_base = step_limit(d_base, MAX_STEP_BASE)
            d_arm1 = step_limit(d_arm1, MAX_STEP_ARM1)

            base += d_base
            arm1 += d_arm1

            base = clamp(base, BASE_MIN, BASE_MAX)
            arm1 = clamp(arm1, ARM1_MIN, ARM1_MAX)

        elif state == GRAB:
            grip = clamp(grip + 3, GRIP_MIN, GRIP_MAX)
            arm1 = clamp(arm1 - 1, ARM1_MIN, ARM1_MAX)

        elif state == PLACE:
            pb, pa1, pa2, pg_open = PLACE_PRESET[target_color]
            t = (time.time() - place_t0) if place_t0 else 0.0
            if t < PLACE_TIME * 0.7:
                base = pb
                arm1 = pa1
                arm2 = pa2
            else:
                grip = pg_open

        base = int(clamp(base, BASE_MIN, BASE_MAX))
        arm1 = int(clamp(arm1, ARM1_MIN, ARM1_MAX))
        arm2 = int(clamp(arm2, ARM2_MIN, ARM2_MAX))
        grip = int(clamp(grip, GRIP_MIN, GRIP_MAX))

        # 전송(주기 제한)
        send_cmd_throttled(ser, base, arm1, arm2, grip)

        # ====== 디버깅 표시 ======
        cv2.circle(frame, (CX0, CY0), 6, (0, 0, 0), 2)
        cv2.putText(frame, f"TARGET:{target_color}  STATE:{state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"CANDS:{len(cands)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        if best is not None:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 0), -1)
            cv2.putText(frame, f"AREA:{int(area)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imshow("camera", frame)
        cv2.imshow("mask", mask)

finally:
    # 종료 시 안전 복귀 + 자원 정리
    try:
        if ser is not None and ser.is_open:
            go_home(ser)
            time.sleep(0.2)
    except Exception:
        pass

    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass

    try:
        if ser is not None and ser.is_open:
            ser.close()
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
