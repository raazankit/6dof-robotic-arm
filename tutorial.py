"""
================================================================================
FINAL MULTI-MODAL HYBRID TELEOPERATION CONTROLLER (DUAL USB VERSION)
================================================================================
Hardware Connections:
  - Robot ESP32: USB Serial (controls 6 MG996R servos via PCA9685)
  - Glove ESP32: USB Serial (streams MPU-6050 IMU + Flex Sensor)
================================================================================
"""

import cv2
import mediapipe as mp
import math
import numpy as np
import serial
import time

# ==============================================================================
# 1. HARDWARE & SERIAL PORT CONFIGURATION
# ==============================================================================
# Adjust to match your ports (check with `ls /dev/cu.usb*` on macOS)
PORT_ROBOT      = '/dev/cu.usbserial-0001'   # ESP32 connected to Robot Arm
PORT_GLOVE      = '/dev/cu.usbserial-8'   # ESP32 wired to Glove Sensors
BAUD_RATE       = 115200
SERIAL_INTERVAL = 0.025                     # 40 Hz serial dispatch (25 ms)

# Connect to Robot ESP32
try:
    robot_ser = serial.Serial(PORT_ROBOT, BAUD_RATE, timeout=0.05)
    robot_ser.dtr = False  # Prevent macOS PySerial from holding ESP32 in reset
    robot_ser.rts = False
    time.sleep(1.0)
    print(f"[✓] Connected to Robot Arm ESP32 on {PORT_ROBOT}")
except Exception as e:
    print(f"[!] Robot Serial Connection Warning: {e}")
    robot_ser = None

# Connect to Glove ESP32
try:
    glove_ser = serial.Serial(PORT_GLOVE, BAUD_RATE, timeout=0.05)
    glove_ser.dtr = False
    glove_ser.rts = False
    time.sleep(1.0)
    print(f"[✓] Connected to Glove ESP32 on {PORT_GLOVE}")
except Exception as e:
    print(f"[!] Glove USB Connection Warning: {e}")
    glove_ser = None


# ==============================================================================
# 2. 1D KALMAN FILTER MODULE
# ==============================================================================
class KalmanFilter1D:
    """Discrete 1D Kalman Filter for fusing accelerometer tilt and gyroscope rate."""
    def __init__(self, q_angle=0.001, q_bias=0.003, r_measure=0.03):
        self.q_angle = q_angle
        self.q_bias = q_bias
        self.r_measure = r_measure
        self.angle = 0.0
        self.bias = 0.0
        self.P = [[1.0, 0.0], [0.0, 1.0]]

    def update(self, new_angle, new_rate, dt):
        dt = max(min(float(dt), 0.1), 0.001)

        # 1. State Prediction
        rate = new_rate - self.bias
        self.angle += dt * rate

        # 2. Covariance Prediction
        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + self.q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.q_bias * dt

        # 3. Measurement Innovation with wrap-around protection
        y = (new_angle - self.angle + 180.0) % 360.0 - 180.0

        # 4. Kalman Gain
        S = self.P[0][0] + self.r_measure
        K = [self.P[0][0] / S, self.P[1][0] / S]

        # 5. State Correction
        self.angle += K[0] * y
        self.bias += K[1] * y

        # 6. Covariance Update with Symmetrization
        p00_temp = self.P[0][0]
        p01_temp = self.P[0][1]
        self.P[0][0] -= K[0] * p00_temp
        self.P[0][1] -= K[0] * p01_temp
        self.P[1][0] -= K[1] * p00_temp
        self.P[1][1] -= K[1] * p01_temp

        p_sym = 0.5 * (self.P[0][1] + self.P[1][0])
        self.P[0][1] = p_sym
        self.P[1][0] = p_sym

        return self.angle


# ==============================================================================
# 3. KINEMATICS ENGINE
# ==============================================================================
class KinematicArm:
    """
    Planar Geometric 2-Link IK with link dimensions in mm:
      - L1: 85 mm  (Base pedestal height)
      - L2: 105 mm (Shoulder link)
      - L3: 130 mm (Elbow link)
      - L4: 170 mm (End-effector claw length)
    """
    def __init__(self, L1=85.0, L2=105.0, L3=130.0, L4=170.0):
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.L3 = float(L3)
        self.L4 = float(L4)

    def solve_ik_2d(self, target_z, target_r):
        D = max(math.hypot(target_r, target_z), 1e-4)

        # Min inner reach clamp (prevents base self-collision)
        min_reach = abs(self.L2 - self.L3) + 20.0
        if D < min_reach:
            scale = min_reach / D
            target_r *= scale
            target_z *= scale
            D = min_reach

        # Max outer reach boundary clamp
        max_reach = (self.L2 + self.L3) * 0.99
        if D > max_reach:
            scale = max_reach / D
            target_r *= scale
            target_z *= scale
            D = max_reach

        # Law of Cosines for elbow angle (theta3)
        cos_theta3 = (D**2 - self.L2**2 - self.L3**2) / (2.0 * self.L2 * self.L3)
        theta3_rad = math.acos(np.clip(cos_theta3, -1.0, 1.0))

        # Shoulder angle (theta2)
        alpha = math.atan2(target_z, target_r)
        beta = math.acos(np.clip((self.L2**2 + D**2 - self.L3**2) / (2.0 * self.L2 * D), -1.0, 1.0))
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
            
        t1, t2, t3, t4 = [a - 90.0 for a in angles[:4]]
        H1 = get_t(t1, self.L1, 0, -90)
        H2 = H1 @ get_t(t2, 0, self.L2, 0)
        H3 = H2 @ get_t(t3, 0, self.L3, 0)
        H4 = H3 @ get_t(t4, 0, self.L4, 0)
        return [np.array([0, 0, 0]), H1[:3, 3], H2[:3, 3], H3[:3, 3], H4[:3, 3]]


# Initialize kinematic arm and Kalman filters
arm = KinematicArm(L1=85.0, L2=105.0, L3=130.0, L4=170.0)
kf_pitch = KalmanFilter1D()
kf_roll  = KalmanFilter1D()


# ==============================================================================
# 4. RUNTIME STATE & MEDIAPIPE INITIALIZATION
# ==============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 0.5 confidence allows tracking hands wearing dark gloves
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

target_servos   = [90, 90, 90, 90, 90, 90]
ema_servos      = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
ema_alpha       = 0.25

# Servo Polarity Vector [0:Base, 1:Shoulder, 2:Elbow, 3:Pitch, 4:Roll, 5:Claw]
invert_axis = [False, False, True, False, False, False]

# Flex Sensor Dynamic Auto-Calibration State
flex_min = 4095
flex_max = 0
FLEX_INVERTED = False  # Set to True if bending the finger DECREASES the ADC reading

last_serial_time  = 0.0
last_glove_time   = 0.0
pitch_offset, roll_offset = None, None
fused_pitch, fused_roll   = 0.0, 0.0
flex_claw_angle   = 0
camera_claw_angle = 0
live_flex_val     = 0
collision_active  = False

print("\n" + "="*75)
print("[*] DUAL-USB TELEOPERATION CONTROLLER ACTIVE")
print("    -> Moving glove updates Wrist & Gripper independently.")
print("    -> Press 'c' to zero-tare the glove IMU.")
print("    -> Press keys '1' through '6' to flip servo polarity.")
print("    -> Press 'q' to safely exit.")
print("="*75 + "\n")


# ==============================================================================
# 5. MAIN EVENT & CONTROL LOOP
# ==============================================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[!] Camera feed interrupted.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()

    # --------------------------------------------------------------------------
    # A. READ GLOVE TELEMETRY OVER USB (DRAIN BUFFER FOR ZERO LAG)
    # --------------------------------------------------------------------------
    latest_glove_line = None
    if glove_ser and glove_ser.is_open:
        # Drain all buffered lines to extract only the freshest reading
        while glove_ser.in_waiting > 0:
            try:
                line = glove_ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    latest_glove_line = line
            except Exception:
                break

    if latest_glove_line:
        parts = latest_glove_line.split(',')
        if len(parts) == 8:
            try:
                ax, ay, az, gx, gy, gz, raw_flex_str, dt = parts
                ax, ay, az = float(ax), float(ay), float(az)
                gx, gy, gz = float(gx), float(gy), float(gz)
                dt = float(dt)
                live_flex_val = int(raw_flex_str)

                # Dynamic Auto-Calibration for Flex Sensor
                if live_flex_val > 100:  # Ignore floating pin noise
                    flex_min = min(flex_min, live_flex_val)
                    flex_max = max(flex_max, live_flex_val)

                span = max(flex_max - flex_min, 60)
                if not FLEX_INVERTED:
                    bend_ratio = (live_flex_val - flex_min) / float(span)
                else:
                    bend_ratio = (flex_max - live_flex_val) / float(span)

                bend_ratio = max(0.0, min(1.0, bend_ratio))
                flex_claw_angle = int(bend_ratio * 180.0)

                # Accelerometer tilt angles via gravity vector
                raw_r = math.atan2(ay, az) * 180.0 / math.pi
                raw_p = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi

                if pitch_offset is None:
                    pitch_offset, roll_offset = raw_p, raw_r

                # Update 1D Kalman state estimators
                fused_pitch = kf_pitch.update(raw_p - pitch_offset, gy, dt)
                fused_roll  = kf_roll.update(raw_r - roll_offset, gx, dt)
                last_glove_time = current_time
            except ValueError:
                pass

    glove_online = (current_time - last_glove_time) < 0.5

    # --------------------------------------------------------------------------
    # B. GLOVE DRIVES WRIST ORIENTATION INDEPENDENTLY (EVERY FRAME)
    # --------------------------------------------------------------------------
    if glove_online:
        raw_pitch_servo = int(np.clip(np.interp(fused_pitch, [-45.0, 45.0], [0, 180]), 0, 180))
        raw_roll_servo  = int(np.clip(np.interp(fused_roll,  [-45.0, 45.0], [0, 180]), 0, 180))
        
        target_servos[3] = (180 - raw_pitch_servo) if invert_axis[3] else raw_pitch_servo
        target_servos[4] = (180 - raw_roll_servo)  if invert_axis[4] else raw_roll_servo

    # --------------------------------------------------------------------------
    # C. VISION TRACKING & DUAL-TRIGGER PINCH CALCULATION
    # --------------------------------------------------------------------------
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        wrist     = hand_lms.landmark[0]
        thumb_tip = hand_lms.landmark[4]
        index_mcp = hand_lms.landmark[5]
        index_tip = hand_lms.landmark[8]
        pinky_mcp = hand_lms.landmark[17]

        # 1. Base Yaw (Servo 0)
        raw_base = int(np.clip(np.interp(wrist.x, [0.1, 0.9], [180, 0]), 0, 180))
        target_servos[0] = (180 - raw_base) if invert_axis[0] else raw_base

        # 2. Reach & Height with Ground Clearance
        palm_len = math.hypot((index_mcp.x - wrist.x) * w, (index_mcp.y - wrist.y) * h)
        palm_wid = math.hypot((index_mcp.x - pinky_mcp.x) * w, (index_mcp.y - pinky_mcp.y) * h) * 1.3
        hand_scale = max(palm_len, palm_wid, 1.0)

        target_r = float(np.interp(hand_scale, [40.0, 140.0], [60.0, 180.0]))
        # Full ground reach down to table surface level (-85 mm)
        raw_z    = float(np.interp(wrist.y, [0.1, 0.9], [160.0, -85.0]))

        # Dynamic ground collision clearance model
        pitch_rad = math.radians(target_servos[3] - 90.0)
        delta_z_drop = arm.L4 * abs(math.sin(pitch_rad))
        z_min_allowed = -arm.L1 + delta_z_drop + 5.0  # 5 mm table buffer
        safe_z = max(raw_z, z_min_allowed)
        collision_active = (raw_z < z_min_allowed)

        # Solve Inverse Kinematics for Shoulder and Elbow (Servos 1 & 2)
        shoulder_ang, elbow_ang = arm.solve_ik_2d(safe_z, target_r)
        s1 = int(np.clip(shoulder_ang, 0, 180))
        s2 = int(np.clip(elbow_ang, 0, 180))

        target_servos[1] = (180 - s1) if invert_axis[1] else s1
        target_servos[2] = (180 - s2) if invert_axis[2] else s2

        # 3. Vision Fallbacks for Wrist (Only if glove is offline)
        if not glove_online:
            vertical_proj = ((wrist.y - index_mcp.y) * h) / max(hand_scale, 1.0)
            raw_pitch = int(np.clip(np.interp(vertical_proj, [-0.6, 0.9], [0, 180]), 0, 180))
            roll_angle = math.degrees(math.atan2((index_mcp.y - pinky_mcp.y) * h, (index_mcp.x - pinky_mcp.x) * w))
            raw_roll = int(np.clip(np.interp(roll_angle, [-45.0, 45.0], [0, 180]), 0, 180))
            
            target_servos[3] = (180 - raw_pitch) if invert_axis[3] else raw_pitch
            target_servos[4] = (180 - raw_roll)  if invert_axis[4] else raw_roll

        # 4. CAMERA PINCH CALCULATION (RUNS CONTINUOUSLY)
        pinch_dist = math.hypot((index_tip.x - thumb_tip.x) * w, (index_tip.y - thumb_tip.y) * h)
        pinch_ratio = pinch_dist / hand_scale
        # Relaxed threshold: ratio <= 0.48 -> 180 deg (closed); ratio >= 0.85 -> 0 deg (open)
        camera_claw_angle = int(np.clip(np.interp(pinch_ratio, [0.48, 0.85], [180, 0]), 0, 180))

        # Visual indicator between thumb and index finger
        tip_thumb_pt = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        tip_index_pt = (int(index_tip.x * w), int(index_tip.y * h))
        pinch_color  = (0, 255, 0) if camera_claw_angle > 100 else (0, 0, 255)
        cv2.line(frame, tip_thumb_pt, tip_index_pt, pinch_color, 3)
    else:
        # Hand not in camera view: camera pinch drops to 0
        camera_claw_angle = 0
        collision_active = False

    # --------------------------------------------------------------------------
    # D. DUAL-TRIGGER GRIPPER ARBITRATION (SERVO 5)
    # --------------------------------------------------------------------------
    # Take whichever input is commanding the claw to close more
    if glove_online:
        raw_claw = max(camera_claw_angle, flex_claw_angle)
    else:
        raw_claw = camera_claw_angle

    target_servos[5] = (180 - raw_claw) if invert_axis[5] else raw_claw

    # --------------------------------------------------------------------------
    # E. CONTINUOUS EMA SMOOTHING (ALL 6 SERVOS)
    # --------------------------------------------------------------------------
    for i in range(6):
        ema_servos[i] = (ema_alpha * target_servos[i]) + ((1.0 - ema_alpha) * ema_servos[i])

    # --------------------------------------------------------------------------
    # F. ISOMETRIC DIGITAL TWIN PROJECTION
    # --------------------------------------------------------------------------
    points_3d = arm.forward_kinematics(ema_servos)
    points_2d = []
    for p in points_3d:
        iso_x = int(120 + (p[0] - p[1] * 0.5) * 0.8)
        iso_y = int(440 - (p[2] - p[1] * 0.3) * 0.8)
        points_2d.append((iso_x, iso_y))

    for i in range(len(points_2d) - 1):
        cv2.line(frame, points_2d[i], points_2d[i + 1], (0, 215, 255), 4, cv2.LINE_AA)
        cv2.circle(frame, points_2d[i], 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, points_2d[-1], 8, (0, 255, 0), -1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # G. DIAGNOSTIC HEADS-UP DISPLAY (HUD)
    # --------------------------------------------------------------------------
    mode_color = (0, 255, 0) if glove_online else (0, 255, 255)
    mode_str = "DUAL-USB: CV + GLOVE (WIRED)" if glove_online else "VISION-ONLY (GLOVE OFFLINE)"
    cv2.putText(frame, f"Mode: {mode_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Servos: {[int(x) for x in ema_servos]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Gripper Diagnostics
    diag_str = f"ADC: {live_flex_val} (Min:{flex_min}, Max:{flex_max}) | Flex: {flex_claw_angle} deg | Cam: {camera_claw_angle} deg -> Claw: {int(ema_servos[5])}"
    cv2.putText(frame, diag_str, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    if collision_active:
        cv2.putText(frame, "[!] COLLISION GUARD: ARM ELEVATED TO PROTECT DESK", (10, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # H. 40 Hz SERIAL DISPATCH TO ROBOT ARM
    # --------------------------------------------------------------------------
    if robot_ser and robot_ser.is_open and (current_time - last_serial_time >= SERIAL_INTERVAL):
        data_packet = f"{int(ema_servos[0])},{int(ema_servos[1])},{int(ema_servos[2])},{int(ema_servos[3])},{int(ema_servos[4])},{int(ema_servos[5])}\n"
        try:
            robot_ser.write(data_packet.encode('utf-8'))
            last_serial_time = current_time
        except serial.SerialException:
            pass

    cv2.imshow("Multi-Modal Dual-USB Teleoperation Engine", frame)

    # --------------------------------------------------------------------------
    # I. KEYBOARD COMMANDS
    # --------------------------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        pitch_offset, roll_offset = None, None
        print("[✓] Glove IMU Zero-Tare Recalibrated!")
    elif ord('1') <= key <= ord('6'):
        idx = key - ord('1')
        invert_axis[idx] = not invert_axis[idx]
        print(f"[*] Toggled Servo[{idx}] Polarity -> {'INVERTED' if invert_axis[idx] else 'NORMAL'}")

# ==============================================================================
# 6. CLEAN SHUTDOWN
# ==============================================================================
print("\n[*] Shutting down teleoperation pipeline...")
cap.release()
cv2.destroyAllWindows()
if glove_ser and glove_ser.is_open:
    glove_ser.close()
    print("[✓] Glove serial port closed.")
if robot_ser and robot_ser.is_open:
    robot_ser.close()
    print("[✓] Robot serial port closed.")
print("[✓] System offline safely.")