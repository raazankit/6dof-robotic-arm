"""
================================================================================
MULTI-MODAL HYBRID TELEOPERATION CONTROLLER FOR 6-DOF ROBOTIC ARM
Features:
  - MediaPipe Monocular Hand Pose Tracking (X, Y, Z translation)
  - MPU-6050 6-Axis IMU with Discrete 1D Kalman Filter (Pitch & Roll)
  - Dual-Trigger Grasp Arbitrator (Vision Pinch + Resistive Flex Sensor)
  - Geometric 2-Link Closed-Form Inverse Kinematics (IK)
  - Active Dynamic Ground Collision Avoidance Guard
  - Two-Stage Exponential Moving Average (EMA) Temporal Smoothing
  - Denavit-Hartenberg Forward Kinematics (FK) Isometric Digital Twin
================================================================================
"""

import cv2
import mediapipe as mp
import math
import numpy as np
import serial
import socket
import time

# ==============================================================================
# 1. HARDWARE & COMMUNICATION CONFIGURATION
# ==============================================================================
PORT_ROBOT      = '/dev/cu.usbserial-0001'  # Robot ESP32 USB Serial Port
BAUD_RATE       = 115200                   # UART Baud Rate
SERIAL_INTERVAL = 0.025                    # Serial dispatch rate: 40 Hz (25 ms)

UDP_IP          = "0.0.0.0"                # Ingest UDP packets on all interfaces
UDP_PORT        = 5005                     # Wireless Glove telemetry port
FLEX_FLAT       = 960                      # Flat finger ADC threshold (0 deg)
FLEX_BENT       = 2000                     # Fully curled finger ADC threshold (180 deg)

# Initialize Serial Connection with macOS DTR/RTS Reset Bypass
try:
    esp32 = serial.Serial(PORT_ROBOT, BAUD_RATE, timeout=0.05)
    esp32.dtr = False  # Prevents macOS PySerial from holding ESP32 in reset
    esp32.rts = False
    time.sleep(1.5)
    print(f"[✓] Connected to Robot Arm ESP32 on {PORT_ROBOT}!")
except Exception as e:
    print(f"[!] Serial Connection Warning: {e}")
    esp32 = None

# Initialize Non-blocking UDP Socket for Wearable Glove Telemetry
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_sock.bind((UDP_IP, UDP_PORT))
    udp_sock.setblocking(False)
    print(f"[✓] UDP Listener active on port {UDP_PORT} (Non-blocking mode)")
except Exception as e:
    print(f"[!] UDP Socket Error: {e}")


# ==============================================================================
# 2. FILTERING & ESTIMATION MODULES (KALMAN & EMA)
# ==============================================================================
class KalmanFilter1D:
    """
    Discrete 1D Kalman Filter for fusing accelerometer tilt and angular rate.
    Includes angle wrap-around protection, covariance symmetrization, and dt clamping.
    """
    def __init__(self, q_angle=0.001, q_bias=0.003, r_measure=0.03):
        self.q_angle = q_angle
        self.q_bias = q_bias
        self.r_measure = r_measure
        self.angle = 0.0
        self.bias = 0.0
        # Initialize error covariance matrix with positive variance
        self.P = [[1.0, 0.0], [0.0, 1.0]]

    def update(self, new_angle, new_rate, dt):
        # Bound dt to eliminate timing spikes caused by Wi-Fi transmission bursts
        dt = max(min(float(dt), 0.1), 0.001)

        # 1. State Prediction (A priori state estimate)
        rate = new_rate - self.bias
        self.angle += dt * rate

        # 2. Error Covariance Extrapolation
        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + self.q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.q_bias * dt

        # 3. Measurement Innovation with +/- 180 deg wrap-around normalization
        y = new_angle - self.angle
        y = (y + 180.0) % 360.0 - 180.0

        # 4. Kalman Gain Computation
        S = self.P[0][0] + self.r_measure
        K = [self.P[0][0] / S, self.P[1][0] / S]

        # 5. State Correction (A posteriori state estimate)
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


class EMAFilter:
    """First-order Exponential Moving Average (EMA) low-pass filter."""
    def __init__(self, alpha=0.25, initial_value=None):
        self.alpha = float(alpha)
        self.value = initial_value

    def update(self, current_value):
        if self.value is None:
            self.value = float(current_value)
        else:
            self.value = (self.alpha * float(current_value)) + ((1.0 - self.alpha) * self.value)
        return self.value

    def reset(self, initial_value=None):
        self.value = initial_value


# ==============================================================================
# 3. KINEMATICS ENGINE (INVERSE & FORWARD KINEMATICS)
# ==============================================================================
class KinematicArm:
    """
    Kinematic model for 6-DOF articulated robotic arm.
    Link dimensions in millimeters:
      - L1: 85 mm  (Base pedestal height to shoulder axis)
      - L2: 105 mm (Shoulder to elbow link)
      - L3: 130 mm (Elbow to wrist link)
      - L4: 170 mm (Wrist to claw grip center)
    """
    def __init__(self, L1=85.0, L2=105.0, L3=130.0, L4=170.0):
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.L3 = float(L3)
        self.L4 = float(L4)

    def solve_ik_2d(self, target_z, target_r):
        """
        Analytical 2-link planar geometric IK with dual-boundary clamping.
        Computes shoulder (theta2) and elbow (theta3) angles.
        """
        D = math.hypot(target_r, target_z)

        # Minimum inner reach clamp (prevents base self-collision & division by zero)
        min_reach = abs(self.L2 - self.L3) + 25.0
        if D < min_reach:
            scale = min_reach / max(D, 1e-4)
            target_r *= scale
            target_z *= scale
            D = min_reach

        # Maximum outer reach boundary clamp (prevents acos domain crash)
        max_reach = (self.L2 + self.L3) * 0.99
        if D > max_reach:
            scale = max_reach / D
            target_r *= scale
            target_z *= scale
            D = max_reach

        # Elbow Angle (theta3) via Law of Cosines
        cos_theta3 = (D**2 - self.L2**2 - self.L3**2) / (2.0 * self.L2 * self.L3)
        theta3_rad = math.acos(np.clip(cos_theta3, -1.0, 1.0))

        # Shoulder Angle (theta2) via planar angle decomposition
        alpha = math.atan2(target_z, target_r)
        cos_beta = (self.L2**2 + D**2 - self.L3**2) / (2.0 * self.L2 * D)
        beta = math.acos(np.clip(cos_beta, -1.0, 1.0))
        theta2_rad = alpha + beta

        return math.degrees(theta2_rad), math.degrees(theta3_rad)

    def forward_kinematics(self, angles):
        """
        Calculates 3D Cartesian coordinates of all revolute joints using DH matrices.
        Used for driving the OpenCV isometric digital twin.
        """
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


# Instantiate Core Kinematic & Filter Engines
arm = KinematicArm(L1=85.0, L2=105.0, L3=130.0, L4=170.0)
kf_pitch = KalmanFilter1D()
kf_roll  = KalmanFilter1D()

# Pre-IK Cartesian EMA Filters (Smoothing input tracking coordinates)
ema_reach  = EMAFilter(alpha=0.30, initial_value=120.0)
ema_height = EMAFilter(alpha=0.30, initial_value=60.0)
ema_base_x = EMAFilter(alpha=0.35, initial_value=0.5)

# Post-IK Output Servo EMA Filters (Smoothing physical motor setpoints)
ema_servo_filters = [EMAFilter(alpha=0.25, initial_value=90.0) for _ in range(6)]


# ==============================================================================
# 4. MEDIAPIPE & RUNTIME STATE INITIALIZATION
# ==============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# Target and filtered servo joint arrays [0:Base, 1:Shoulder, 2:Elbow, 3:Pitch, 4:Roll, 5:Claw]
target_servos   = [90, 90, 90, 90, 90, 90]
smoothed_servos = [90, 90, 90, 90, 90, 90]

# Servo Direction Polarity Vector (Matches physical bracket mounting)
invert_axis = [False, False, True, True, False, False]

# Runtime telemetry tracking variables
last_serial_time = 0.0
last_glove_time  = 0.0
pitch_offset, roll_offset = None, None
fused_pitch, fused_roll   = 0.0, 0.0
flex_claw_angle   = 0
camera_claw_angle = 0
collision_active  = False

print("\n" + "="*70)
print("[*] 6-DOF HYBRID TELEOPERATION SYSTEM ACTIVE")
print("    -> Press keys '1' through '6' to flip servo inversion live.")
print("    -> Press 'c' to re-zero / tare the glove IMU orientation.")
print("    -> Press 'q' to safely exit.")
print("="*70 + "\n")


# ==============================================================================
# 5. MAIN EVENT & CONTROL LOOP
# ==============================================================================
# Auto-calibrating Flex Sensor Tracker
flex_min = 4095
flex_max = 0
FLEX_INVERTED = False  # Set to True if bending the finger DECREASES the ADC reading
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[!] Video capture frame dropped or camera disconnected.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()

    # --------------------------------------------------------------------------
    # A. INGEST WEARABLE GLOVE TELEMETRY (UDP BUFFER DRAIN)
    # --------------------------------------------------------------------------
    latest_udp = None
    while True:
        try:
            data, _ = udp_sock.recvfrom(1024)
            latest_udp = data  # Drain socket to extract freshest telemetry packet
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

                # Accelerometer tilt angles via gravity vector decomposition
                raw_r = math.atan2(ay, az) * 180.0 / math.pi
                raw_p = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi

                # Auto-tare on first valid glove packet
                if pitch_offset is None:
                    pitch_offset, roll_offset = raw_p, raw_r

                # Update 1D Kalman state estimators
                fused_pitch = kf_pitch.update(raw_p - pitch_offset, gy, dt)
                fused_roll  = kf_roll.update(raw_r - roll_offset, gx, dt)

                # Map flex sensor ADC to claw closure
                flex_claw_angle = int(np.clip(np.interp(live_flex, [FLEX_FLAT, FLEX_BENT], [0, 180]), 0, 180))
                last_glove_time = current_time
            except ValueError:
                pass

    # Glove considered online if packet arrived within the last 500 ms
    glove_online = (current_time - last_glove_time) < 0.5

    # --------------------------------------------------------------------------
    # B. COMPUTER VISION & KINEMATICS PIPELINE
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

        # 1. Base Yaw (Servo 0) - Linear mapping of screen X position
        smooth_wrist_x = ema_base_x.update(wrist.x)
        raw_base = int(np.clip(np.interp(smooth_wrist_x, [0.1, 0.9], [180, 0]), 0, 180))
        target_servos[0] = (180 - raw_base) if invert_axis[0] else raw_base

        # 2. Wrist Orientation (Pitch & Roll - Servos 3 & 4)
        if glove_online:
            raw_pitch_servo = int(np.clip(np.interp(fused_pitch, [-45.0, 45.0], [0, 180]), 0, 180))
            raw_roll_servo  = int(np.clip(np.interp(fused_roll, [-45.0, 45.0], [0, 180]), 0, 180))
        else:
            hand_scale_temp = math.hypot((index_mcp.x - wrist.x) * w, (index_mcp.y - wrist.y) * h)
            vertical_proj = ((wrist.y - index_mcp.y) * h) / max(hand_scale_temp, 1.0)
            raw_pitch_servo = int(np.clip(np.interp(vertical_proj, [-0.6, 0.9], [0, 180]), 0, 180))

            roll_angle = math.degrees(math.atan2((index_mcp.y - pinky_mcp.y) * h, (index_mcp.x - pinky_mcp.x) * w))
            raw_roll_servo = int(np.clip(np.interp(roll_angle, [-45.0, 45.0], [0, 180]), 0, 180))

        target_servos[3] = (180 - raw_pitch_servo) if invert_axis[3] else raw_pitch_servo
        target_servos[4] = (180 - raw_roll_servo)  if invert_axis[4] else raw_roll_servo

        # 3. Dynamic Ground Collision Clearance Enforcement
        # Converts pitch servo angle to downward angular dip relative to horizontal (90 deg)
        pitch_rad = math.radians(raw_pitch_servo - 90.0)
        delta_z_drop = arm.L4 * abs(math.sin(pitch_rad))
        z_clearance = 35.0  # Safe air gap above the table plane (mm)
        z_min_allowed = -arm.L1 + delta_z_drop + z_clearance

        # Hand elevation to vertical Cartesian height target
        hand_scale = math.hypot((index_mcp.x - wrist.x) * w, (index_mcp.y - wrist.y) * h)
        raw_r = float(np.interp(hand_scale, [40.0, 140.0], [60.0, 180.0]))
        raw_z = float(np.interp(wrist.y, [0.1, 0.9], [160.0, -20.0]))

        # Enforce dynamic boundary floor
        safe_z = max(raw_z, z_min_allowed)
        collision_active = (raw_z < z_min_allowed)

        # Pre-IK Cartesian EMA Smoothing
        target_r = ema_reach.update(raw_r)
        target_z = ema_height.update(safe_z)

        # 4. Shoulder & Elbow Angles via Inverse Kinematics (Servos 1 & 2)
        shoulder_ang, elbow_ang = arm.solve_ik_2d(target_z, target_r)
        s1 = int(np.clip(shoulder_ang, 0, 180))
        s2 = int(np.clip(elbow_ang, 0, 180))

        target_servos[1] = (180 - s1) if invert_axis[1] else s1
        target_servos[2] = (180 - s2) if invert_axis[2] else s2

        # 5. Dual-Trigger Gripper Arbitration (Servo 5)
        pinch_dist = math.hypot((index_tip.x - thumb_tip.x) * w, (index_tip.y - thumb_tip.y) * h)
        pinch_ratio = pinch_dist / max(hand_scale, 1.0)
        camera_claw_angle = int(np.clip(np.interp(pinch_ratio, [0.35, 0.85], [180, 0]), 0, 180))

        raw_claw = max(camera_claw_angle, flex_claw_angle) if glove_online else camera_claw_angle
        target_servos[5] = (180 - raw_claw) if invert_axis[5] else raw_claw

        # 6. Post-IK Servo Low-Pass EMA Smoothing
        for i in range(6):
            smoothed_servos[i] = int(round(ema_servo_filters[i].update(target_servos[i])))

    # --------------------------------------------------------------------------
    # C. ISOMETRIC DIGITAL TWIN PROJECTION (OPENCV)
    # --------------------------------------------------------------------------
    points_3d = arm.forward_kinematics(smoothed_servos)
    points_2d = []
    
    # Axonometric isometric projection transformation
    for p in points_3d:
        iso_x = int(120 + (p[0] - p[1] * 0.5) * 0.8)
        iso_y = int(440 - (p[2] - p[1] * 0.3) * 0.8)
        points_2d.append((iso_x, iso_y))

    # Render digital twin kinematic bones and revolute joint vertices
    for i in range(len(points_2d) - 1):
        cv2.line(frame, points_2d[i], points_2d[i + 1], (0, 215, 255), 4, cv2.LINE_AA)
        cv2.circle(frame, points_2d[i], 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, points_2d[-1], 8, (0, 255, 0), -1, cv2.LINE_AA)  # End-effector tip

    # --------------------------------------------------------------------------
    # D. ON-SCREEN HEADS-UP DISPLAY (HUD)
    # --------------------------------------------------------------------------
    mode_color = (0, 255, 0) if glove_online else (0, 255, 255)
    mode_str = "HYBRID: CV + KALMAN GLOVE" if glove_online else "VISION-ONLY (GLOVE OFFLINE)"
    cv2.putText(frame, f"System Mode: {mode_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Smoothed Servos: {smoothed_servos}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Invert Flags (Keys 1-6): {['INV' if x else 'NRM' for x in invert_axis]}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if collision_active:
        cv2.putText(frame, "[!] COLLISION GUARD ACTIVE: HEIGHT ELEVATED", (10, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # E. RATE-LIMITED SERIAL DISPATCH (40 Hz)
    # --------------------------------------------------------------------------
    if esp32 and esp32.is_open and (current_time - last_serial_time >= SERIAL_INTERVAL):
        data_packet = f"{smoothed_servos[0]},{smoothed_servos[1]},{smoothed_servos[2]},{smoothed_servos[3]},{smoothed_servos[4]},{smoothed_servos[5]}\n"
        try:
            esp32.write(data_packet.encode('utf-8'))
            last_serial_time = current_time
        except serial.SerialException:
            pass

    # Display composite preview
    cv2.imshow("Multi-Modal Hybrid Teleoperation Engine", frame)

    # --------------------------------------------------------------------------
    # F. USER KEYBOARD INTERRUPTS
    # --------------------------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        pitch_offset, roll_offset = None, None
        print("[✓] Wearable Glove IMU Zero-Tare Recalibrated!")
    elif ord('1') <= key <= ord('6'):
        idx = key - ord('1')
        invert_axis[idx] = not invert_axis[idx]
        print(f"[*] Toggled Servo[{idx}] Polarity -> {'INVERTED' if invert_axis[idx] else 'NORMAL'}")

# ==============================================================================
# 6. GRACEFUL SHUTDOWN
# ==============================================================================
print("\n[*] Shutting down teleoperation pipeline...")
cap.release()
cv2.destroyAllWindows()
udp_sock.close()
if esp32 and esp32.is_open:
    esp32.close()
    print("[✓] Serial port released.")
print("[✓] System offline safely.")