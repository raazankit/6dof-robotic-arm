import serial
import time
import math
import numpy as np

# --- 1D KALMAN FILTER CLASS ---
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

        P00_temp = self.P[0][0]
        P01_temp = self.P[0][1]

        self.P[0][0] -= K[0] * P00_temp
        self.P[0][1] -= K[0] * P01_temp
        self.P[1][0] -= K[1] * P00_temp
        self.P[1][1] -= K[1] * P01_temp

        return self.angle

# --- SERIAL CONNECTION ---
# Update to your serial port
SERIAL_PORT = '/dev/cu.usbserial-0001'
BAUD_RATE = 115200

kf_pitch = KalmanFilter1D()
kf_roll = KalmanFilter1D()

# Offset baselines
pitch_offset = 0.0
roll_offset = 0.0

def calibrate_sensors(ser, samples=100):
    print("\n[!] Keep hand completely FLAT and STILL on table for calibration...")
    collected = 0
    p_sum, r_sum = 0.0, 0.0
    
    while collected < samples:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        parts = line.split(',')
        if len(parts) == 7:
            try:
                ax, ay, az, gx, gy, gz, dt = map(float, parts)
                raw_roll = math.atan2(ay, az) * 180.0 / math.pi
                raw_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi
                p_sum += raw_pitch
                r_sum += raw_roll
                collected += 1
            except ValueError:
                pass
                
    print("[✓] Calibration Complete! Neutral hand is now (0°, 0°).\n")
    return (p_sum / samples), (r_sum / samples)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for ESP32 boot
    
    # Run automatic zeroing
    pitch_offset, roll_offset = calibrate_sensors(ser, samples=80)

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) == 7:
            try:
                ax, ay, az, gx, gy, gz, dt = map(float, parts)

                # Raw angles relative to gravity
                accel_roll = (math.atan2(ay, az) * 180.0 / math.pi) - roll_offset
                accel_pitch = (math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi) - pitch_offset

                # Kalman state estimation
                fused_pitch = kf_pitch.update(accel_pitch, gy, dt)
                fused_roll = kf_roll.update(accel_roll, gx, dt)

                # ==========================================
                # MAPPING TO ROBOTIC ARM TARGETS
                # ==========================================
                # 1. Wrist Roll -> Base Servo (0° to 180°, centered at 90°)
                # Hand tilted left (-45°) -> Base at 45°
                # Hand tilted right (+45°) -> Base at 135°
                base_angle = int(np.interp(fused_roll, [-45.0, 45.0], [45, 135]))

                # 2. Hand Pitch -> Reach / Height (r, z)
                # Hand tilted up (+30°) -> Arm reaches higher
                # Hand tilted down (-30°) -> Arm dips down
                target_z = float(np.interp(fused_pitch, [-30.0, 30.0], [-20.0, 80.0]))
                target_r = 110.0  # Constant nominal forward reach in mm

                print(f"Hand -> Pitch: {fused_pitch:5.1f}° | Roll: {fused_roll:5.1f}°  ==>  Robot -> Base: {base_angle}° | Z-Target: {target_z:4.1f}mm")

            except ValueError:
                pass

except KeyboardInterrupt:
    print("\nStopping.")
    if 'ser' in locals() and ser.is_open:
        ser.close()