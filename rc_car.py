import cv2
import numpy as np
import requests
import csv
import time
from datetime import datetime

# Camera stream and sensor endpoints
BASE_URL = "http://10.55.223.213:8080"
STREAM_URL = f"{BASE_URL}/video"
SENSOR_URL = f"{BASE_URL}/sensors.json"

# Log file setup
CSV_FILE = "crack_log.csv"
with open(CSV_FILE, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header if file is empty
    if file.tell() == 0:
        writer.writerow(["Timestamp", "Severity", "Length_px", "Avg_Width_px", "Latitude", "Longitude"])

# Cooldown tracker to prevent writing duplicate logs for the same crack
last_log_time = 0
LOG_COOLDOWN = 3.0  # seconds

def get_gps_coordinates():
    """Fetch live GPS data from the phone's IP Webcam sensor endpoint."""
    try:
        resp = requests.get(SENSOR_URL, timeout=0.3)
        if resp.status_code == 200:
            data = resp.json()
            if "gps" in data and "data" in data["gps"]:
                gps_vals = data["gps"]["data"]
                # Format: [lat, lon, alt, speed, accuracy]
                if gps_vals and len(gps_vals[-1][1]) >= 2:
                    lat = gps_vals[-1][1][0]
                    lon = gps_vals[-1][1][1]
                    return lat, lon
    except Exception:
        pass
    return "No_Fix", "No_Fix"

print("Connecting to IP Webcam stream...")
cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(frame, (480, 360))
    h, w, _ = frame.shape

    # Focus on the road/ground area (bottom 50% of the frame)
    roi_top = int(h * 0.5)
    roi = frame[roi_top:h, 0:w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. Morphological Black-Hat Filter: Extracts dark elements on a lighter background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 2. Thresholding: Isolate high-contrast fissure lines
    blurred = cv2.GaussianBlur(blackhat, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # Morphological closing to bridge small breaks in the crack line
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    # 3. Geometric Crack Analysis via Contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crack_detected = False
    detected_length = 0
    detected_width = 0
    severity = "None"

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:  # Ignore tiny noise / aggregate specks
            continue

        # Fit minimum area rectangle to measure major/minor axis
        rect = cv2.minAreaRect(cnt)
        (center), (rect_w, rect_h), angle = rect
        length = max(rect_w, rect_h)
        width_px = min(rect_w, rect_h)

        # Aspect Ratio check: Cracks are narrow and elongated
        aspect_ratio = length / (width_px + 1e-5)

        if aspect_ratio >= 2.8 and length > 25:
            crack_detected = True
            detected_length = int(length)
            detected_width = round(width_px, 1)

            # Classify severity
            if width_px >= 4.0 or length > 120:
                severity = "Severe"
                color = (0, 0, 255)      # Red
            elif width_px >= 2.0:
                severity = "Moderate"
                color = (0, 165, 255)    # Orange
            else:
                severity = "Minor"
                color = (0, 255, 255)    # Yellow

            # Draw the detected crack on the main display
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            # Offset box back to frame coordinates
            box[:, 1] += roi_top
            cv2.drawContours(frame, [box], 0, color, 2)

            # Annotation text
            cv2.putText(frame, f"{severity} ({detected_length}px)", 
                        (box[0][0], max(20, box[0][1] - 8)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            break  # Track the primary crack in view

    # 4. GPS Pinning and Logging
    current_time = time.time()
    if crack_detected and (current_time - last_log_time > LOG_COOLDOWN):
        lat, lon = get_gps_coordinates()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, severity, detected_length, detected_width, lat, lon])

        print(f"[{timestamp}] Pinned Crack: {severity} | L={detected_length}px W={detected_width}px | GPS: ({lat}, {lon})")
        last_log_time = current_time

    # 5. Dashboard Overlay
    cv2.rectangle(frame, (0, roi_top), (w, h), (255, 255, 0), 1)
    status_msg = f"Status: {severity} Crack" if crack_detected else "Status: Clear Surface"
    hud_color = (0, 0, 255) if crack_detected else (0, 255, 0)
    cv2.putText(frame, status_msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)

    cv2.imshow("Crack Analyzer & GPS Mapper", frame)
    cv2.imshow("Filtered Fissure Mask", cleaned)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()