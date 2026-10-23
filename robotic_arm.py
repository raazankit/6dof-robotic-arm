import cv2
import mediapipe as mp
import math
import numpy as np
import serial
import socket
import time

# ==========================================
# 1. HARDWARE & SERIAL SETUP
# ==========================================
PORT_ROBOT = '/dev/cu.usbserial-0001'
BAUD_RATE  = 115200

# Optional Wireless Glove UDP Port (Auto-detected if glove is powered on)
UDP_IP   = "0.0.0.0"
UDP_PORT = 5005
FLEX_FLAT = 960
FLEX_BENT = 2000

try:
    esp32 = serial.Serial(PORT_ROBOT, BAUD_RATE, timeout=0.05)
    esp32.dtr = False  # Prevents macOS PySerial from holding ESP32 in reset
    esp32.rts = False
    time.sleep(2)
    print("[✓] Connected to Robot Arm ESP32!")
except Exception as e:
    print(f"[!] Serial Connection Failed: {e}")
    esp32 = None

# Non-blocking UDP socket for optional Glove IMU + Flex Sensor
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.bind((UDP_IP, UDP_PORT))
    udp_sock.setblocking(False)
except Exception as e:
    print(f"[!] UDP Socket Warning: {e}")

# ==========================================
# 2. 1D KALMAN FILTER (FOR GLOVE IMU FUSION)
# ==========================================
class KalmanFilter1D:
    def __init__(self, q_angle=0.001, q_bias=0.003, r_measure=0.03):
        self.q_angle = q_angle
        self.q_bias = q_bias
        self.r_measure = r_measure
        self.angle = 0.0
        self.bias = 0.0
        self.P = [[0.0, 0.0], [0.0, 0.0]]

    def update(self, new_angle, new_rate, dt):
        rate = new_rate - self.bias
        self.angle += dt * rate

        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + self.q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.q_bias * dt

        y = new_angle - self.angle
        S = self.P[0][0] + self.r_measure
        K = [self.P[0][0] / S, self.P[1][0] / S]

        self.angle += K[0] * y
        self.bias += K[1] * y

        P00_temp, P01_temp = self.P[0][0], self.P[0][1]
        self.P[0][0] -= K[0] * P00_temp
        self.P[0][1] -= K[0] * P01_temp
        self.P[1][0] -= K[1] * P00_temp
        self.P[1][1] -= K[1] * P01_temp

        return self.angle

# ==========================================
# 3. KINEMATICS ENGINE
# ==========================================
class KinematicArm:
    def __init__(self):
        self.L1 = 40  
        self.L2 = 80  
        self.L3 = 80  
        self.L4 = 40  

    def solve_ik_2d(self, target_z, target_r):
        D = max(math.hypot(target_r, target_z), 1e-5)
        max_reach = self.L2 + self.L3
        
        if D > max_reach:
            scale = (max_reach / D) * 0.99
            target_r *= scale
            target_z *= scale
            D = math.hypot(target_r, target_z)

        cos_theta3 = (D**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        theta3_rad = math.acos(np.clip(cos_theta3, -1.0, 1.0))
        
        alpha = math.atan2(target_z, target_r)
        beta = math.acos(np.clip((self.L2**2 + D**2 - self.L3**2) / (2 * self.L2 * D), -1.0, 1.0))
        theta2_rad = alpha + beta

        return math.degrees(theta2_rad), math.degrees(theta3_rad)

    def forward_kinematics(self, angles):
        def get_t(theta, d, a, alpha_angle):
            th, al = np.radians(theta), np.radians(alpha_angle)
            return np.array([
                [np.cos(th), -np.sin(th)*np.cos(al),  np.sin(th)*np.sin(al), a*np.cos(th)],
                [np.sin(th),  np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
                [0,           np.sin(al),             np.cos(al),            d],
                [0,           0,                      0,                     1]
            ])
            
        t1, t2, t3, t4 = [a - 90 for a in angles[:4]]
        
        H1 = get_t(t1, self.L1, 0, -90)
        H2 = H1 @ get_t(t2, 0, self.L2, 0)
        H3 = H2 @ get_t(t3, 0, self.L3, 0)
        H4 = H3 @ get_t(t4, 0, self.L4, 0)
        
        return [np.array([0,0,0]), H1[:3,3], H2[:3,3], H3[:3,3], H4[:3,3]]

arm = KinematicArm()
kf_pitch = KalmanFilter1D()
kf_roll  = KalmanFilter1D()

# ==========================================
# 4. MEDIAPIPE & STATE TUNING
# ==========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

target_servos = [90, 90, 90, 90, 90, 90]
ema_servos = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
ema_alpha = 0.25

# Updated Default Polarity: 2nd (Shoulder) -> False, 4th (Wrist Pitch) -> True
# [1:Base, 2:Shoulder, 3:Elbow, 4:WristPitch, 5:WristRoll, 6:Claw]
invert_axis = [False, False, True, True, False, False]

last_serial_time = 0.0
SERIAL_INTERVAL = 0.025  # 40 Hz

# Glove IMU states (Auto-activates when UDP packets arrive)
last_glove_time = 0.0
pitch_offset, roll_offset = None, None
fused_pitch, fused_roll = 0.0, 0.0
flex_claw_angle = 0
camera_claw_angle = 0

print("\n[*] Teleoperation Active!")
print("    -> Press keys '1'-'6' to flip any servo direction live.")
print("    -> Press 'c' to re-zero the glove IMU.")
print("    -> Press 'q' to quit.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        print("CRASH REASON: Camera frame read failed.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()

    # ====================================================
    # 1. OPTIONAL GLOVE STREAM (AUTO-FALLBACK IF OFF)
    # ====================================================
    latest_udp = None
    while True:
        try:
            data, _ = udp_sock.recvfrom(1024)
            latest_udp = data
        except BlockingIOError:
            break
        except Exception:
            break

    if latest_udp:
        parts = latest_udp.decode('utf-8', errors='ignore').strip().split(',')
        if len(parts) == 8:
            try:
                ax, ay, az, gx, gy, gz, raw_flex_str, dt = parts
                ax, ay, az = float(ax), float(ay), float(az)
                gx, gy, gz = float(gx), float(gy), float(gz)
                dt = float(dt)
                live_flex = int(raw_flex_str)

                raw_r = math.atan2(ay, az) * 180.0 / math.pi
                raw_p = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi

                # Auto-tare on first packet
                if pitch_offset is None:
                    pitch_offset, roll_offset = raw_p, raw_r

                fused_pitch = kf_pitch.update(raw_p - pitch_offset, gy, dt)
                fused_roll  = kf_roll.update(raw_r - roll_offset, gx, dt)
                flex_claw_angle = int(np.clip(np.interp(live_flex, [FLEX_FLAT, FLEX_BENT], [0, 180]), 0, 180))
                last_glove_time = current_time
            except ValueError:
                pass

    glove_online = (current_time - last_glove_time) < 0.5

    # ====================================================
    # 2. MEDIAPIPE COMPUTER VISION PIPELINE
    # ====================================================
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        wrist     = hand_lms.landmark[0]
        thumb_tip = hand_lms.landmark[4]
        index_mcp = hand_lms.landmark[5]
        index_tip = hand_lms.landmark[8]
        pinky_mcp = hand_lms.landmark[17]

        # A. Base Yaw (1st Servo)
        raw_base = int(np.clip(np.interp(wrist.x, [0.1, 0.9], [180, 0]), 0, 180))
        target_servos[0] = (180 - raw_base) if invert_axis[0] else raw_base

        # B. Reach (target_r) & Height (target_z) — (2nd & 3rd Servos)
        hand_scale = math.hypot((index_mcp.x - wrist.x) * w, (index_mcp.y - wrist.y) * h)
        target_r = float(np.interp(hand_scale, [40.0, 140.0], [50.0, 130.0]))
        target_z = float(np.interp(wrist.y, [0.1, 0.9], [130.0, -10.0]))

        shoulder_ang, elbow_ang = arm.solve_ik_2d(target_z, target_r)
        s1 = int(np.clip(shoulder_ang, 0, 180))
        s2 = int(np.clip(elbow_ang, 0, 180))

        target_servos[1] = (180 - s1) if invert_axis[1] else s1
        target_servos[2] = (180 - s2) if invert_axis[2] else s2

        # C. Wrist Pitch (4th Servo) & Wrist Roll (5th Servo)
        if glove_online:
            raw_pitch_servo = int(np.clip(np.interp(fused_pitch, [-45.0, 45.0], [0, 180]), 0, 180))
            raw_roll_servo  = int(np.clip(np.interp(fused_roll, [-45.0, 45.0], [0, 180]), 0, 180))
        else:
            vertical_proj = ((wrist.y - index_mcp.y) * h) / max(hand_scale, 1.0)
            raw_pitch_servo = int(np.clip(np.interp(vertical_proj, [-0.6, 0.9], [0, 180]), 0, 180))

            roll_angle = math.degrees(math.atan2((index_mcp.y - pinky_mcp.y) * h, (index_mcp.x - pinky_mcp.x) * w))
            raw_roll_servo = int(np.clip(np.interp(roll_angle, [-45.0, 45.0], [0, 180]), 0, 180))

        target_servos[3] = (180 - raw_pitch_servo) if invert_axis[3] else raw_pitch_servo
        target_servos[4] = (180 - raw_roll_servo)  if invert_axis[4] else raw_roll_servo

        # D. Gripper (6th Servo)
        pinch_dist = math.hypot((index_tip.x - thumb_tip.x) * w, (index_tip.y - thumb_tip.y) * h)
        pinch_ratio = pinch_dist / max(hand_scale, 1.0)
        camera_claw_angle = int(np.clip(np.interp(pinch_ratio, [0.35, 0.85], [180, 0]), 0, 180))

        raw_claw = max(camera_claw_angle, flex_claw_angle) if glove_online else camera_claw_angle
        target_servos[5] = (180 - raw_claw) if invert_axis[5] else raw_claw

        # Apply EMA smoothing across all 6 joints
        for i in range(6):
            ema_servos[i] = (ema_alpha * target_servos[i]) + ((1 - ema_alpha) * ema_servos[i])
    
    # ====================================================
    # 3. RENDER ISOMETRIC DIGITAL TWIN
    # ====================================================
    points_3d = arm.forward_kinematics(ema_servos) 
    points_2d = []
    
    for p in points_3d:
        iso_x = int(150 + (p[0] - p[1] * 0.5) * 2.5)
        iso_y = int(450 - (p[2] - p[1] * 0.3) * 2.5)
        points_2d.append((iso_x, iso_y))

    for i in range(len(points_2d) - 1):
        cv2.line(frame, points_2d[i], points_2d[i + 1], (0, 200, 255), 4)
        cv2.circle(frame, points_2d[i], 6, (0, 0, 255), -1)

    # UI Feedback
    mode_str = "HYBRID (CV + KALMAN GLOVE)" if glove_online else "VISION ONLY (GLOVE OFF)"
    cv2.putText(frame, f"Mode: {mode_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Raw Targets: {target_servos}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"EMA Smoothed: {[int(x) for x in ema_servos]}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Invert Flags (Keys 1-6): {['INV' if x else 'NRM' for x in invert_axis]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ====================================================
    # 4. RATE-LIMITED SERIAL TRANSMISSION (40 Hz)
    # ====================================================
    if esp32 and esp32.is_open and (current_time - last_serial_time >= SERIAL_INTERVAL):
        data_packet = f"{int(ema_servos[0])},{int(ema_servos[1])},{int(ema_servos[2])},{int(ema_servos[3])},{int(ema_servos[4])},{int(ema_servos[5])}\n"
        esp32.write(data_packet.encode('utf-8'))
        last_serial_time = current_time
        
    cv2.imshow("IK Teleoperation Visualizer", frame)
    
    # Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c'):
        pitch_offset, roll_offset = None, None
        print("[✓] Glove IMU Zero-Tare Reset!")
    elif ord('1') <= key <= ord('6'):
        idx = key - ord('1')
        invert_axis[idx] = not invert_axis[idx]
        print(f"[*] Toggled Servo[{idx}] Inversion -> {invert_axis[idx]}")

cap.release()
cv2.destroyAllWindows()
udp_sock.close()
if esp32 and esp32.is_open:
    esp32.close()