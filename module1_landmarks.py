import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip image for natural mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Extract normalized landmarks (0.0 to 1.0)
        lms = results.multi_hand_landmarks[0].landmark

        # Convert normalized coordinates to actual screen pixel coordinates
        wrist = (int(lms[0].x * w), int(lms[0].y * h))
        thumb_tip = (int(lms[4].x * w), int(lms[4].y * h))
        index_mcp = (int(lms[5].x * w), int(lms[5].y * h))
        index_tip = (int(lms[8].x * w), int(lms[8].y * h))
        pinky_mcp = (int(lms[17].x * w), int(lms[17].y * h))

        # Visual feedback: Draw colored circles on these 5 critical points
        cv2.circle(frame, wrist, 8, (0, 0, 255), -1)      # Red = Wrist
        cv2.circle(frame, thumb_tip, 8, (255, 0, 0), -1)   # Blue = Thumb Tip
        cv2.circle(frame, index_mcp, 8, (0, 255, 0), -1)  # Green = Index Knuckle
        cv2.circle(frame, index_tip, 8, (255, 255, 0), -1) # Cyan = Index Tip
        cv2.circle(frame, pinky_mcp, 8, (255, 0, 255), -1)# Magenta = Pinky Knuckle

        # On-screen display
        cv2.putText(frame, f"Wrist Screen Pos: {wrist}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Module 1: Critical Landmark Extractor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()