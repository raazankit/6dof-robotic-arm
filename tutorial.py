import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,              # Track only 1 hand for low latency
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 2. Start Webcam Stream
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Camera feed unavailable.")
        break

    # Flip image horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR (OpenCV standard) to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to detect hands
    results = hands.process(rgb_frame)

    # If a hand is detected in the frame:
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the 21 dots and connecting lines over your hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the final output window
    cv2.imshow("Robotic Arm Vision Control - Hand Tracker", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()