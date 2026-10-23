import cv2
import mediapipe as mp
import math
import numpy as np
import serial
import time

# --- Serial Setup ---
try:
    esp32 = serial.Serial('/dev/cu.usbserial-0001', 115200, timeout=0.05)
    time.sleep(2)
    print("Connected to ESP32!")
except Exception as e:
    print(f"Serial Connection Failed: {e}")
    esp32 = None

# --- Kinematics Engine ---
class KinematicArm:
    def __init__(self):
        self.L1 = 40  
        self.L2 = 80  
        self.L3 = 80  
        self.L4 = 40  

    def solve_ik_2d(self, target_z, target_r):
        D = math.hypot(target_r, target_z)
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

# Instantiate once outside the loop
arm = KinematicArm()

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# --- State Arrays & Tuning ---
target_servos = [90, 90, 90, 90, 90, 90]
ema_servos = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
ema_alpha = 0.25

# Rate-limiting serial communication to 40 Hz (every 25ms)
last_serial_time = 0.0
SERIAL_INTERVAL = 0.025  

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        print("CRASH REASON: Camera frame read failed.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        wrist = hand_lms.landmark[0]
        thumb_tip = hand_lms.landmark[4]
        index_mcp = hand_lms.landmark[5]
        index_tip = hand_lms.landmark[8]
        pinky_mcp = hand_lms.landmark[17]

        # 1. Base Yaw (Full frame width coverage: 0.1 to 0.9)
        target_servos[0] = int(np.clip(np.interp(wrist.x, [0.1, 0.9], [180, 0]), 0, 180))

        # 2. Reach (target_r) & Height (target_z)
        # Use apparent hand scale (wrist to middle knuckle distance) for stable forward/back depth
        hand_scale = math.hypot((index_mcp.x - wrist.x) * w, (index_mcp.y - wrist.y) * h)
        # hand_scale ~ 40px (far away) to 140px (close to camera)
        target_r = np.interp(hand_scale, [40, 140], [50, 130])

        # Vertical height: map wrist.y smoothly across the screen height
        target_z = np.interp(wrist.y, [0.9, 0.1], [-10, 130])

        shoulder_ang, elbow_ang = arm.solve_ik_2d(target_z, target_r)
        target_servos[1] = int(np.clip(shoulder_ang, 0, 180))
        target_servos[2] = int(np.clip(180 - elbow_ang, 0, 180))

        # 3. Wrist Pitch (Relative to arm angle or vertical)
        dy = (index_mcp.y - wrist.y) * h
        dx = (index_mcp.x - wrist.x) * w
        hand_pitch = math.degrees(math.atan2(dy, dx))
        # Maps pointing down (-90) to pointing up (+90) into 0-180 servo degrees
        # Inverted Wrist Pitch: Swapped [0, 180] -> [180, 0]
        target_servos[3] = int(np.clip(np.interp(hand_pitch, [-135, -45], [180, 0]), 0, 180))

        # 4. Wrist Roll (Tilt across palm knuckles)
        roll_angle = math.degrees(math.atan2((index_mcp.y - pinky_mcp.y) * h, (index_mcp.x - pinky_mcp.x) * w))
        target_servos[4] = int(np.clip(np.interp(roll_angle, [-45, 45], [0, 180]), 0, 180))

        # 5. Gripper (Pinch distance)
        pinch_dist = math.hypot((index_tip.x - thumb_tip.x) * w, (index_tip.y - thumb_tip.y) * h)
        target_servos[5] = int(np.clip(np.interp(pinch_dist, [25, 110], [0, 180]), 0, 180))

        # Apply EMA smoothing across all active joints
        for i in range(6):
            ema_servos[i] = (ema_alpha * target_servos[i]) + ((1 - ema_alpha) * ema_servos[i])
    
    # --- Render Isometric Digital Twin ---
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
    cv2.putText(frame, f"Raw Targets: {target_servos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"EMA Smoothed: {[int(x) for x in ema_servos]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # --- Rate-Limited Serial Transmission ---
    current_time = time.time()
    if esp32 and esp32.is_open and (current_time - last_serial_time >= SERIAL_INTERVAL):
        data_packet = f"{int(ema_servos[0])},{int(ema_servos[1])},{int(ema_servos[2])},{int(ema_servos[3])},{int(ema_servos[4])},{int(ema_servos[5])}\n"
        esp32.write(data_packet.encode('utf-8'))
        last_serial_time = current_time
        
    cv2.imshow("IK Teleoperation Visualizer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
if esp32 and esp32.is_open:
    esp32.close()