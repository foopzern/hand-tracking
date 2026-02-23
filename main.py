import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,        # False for video stream
    max_num_hands=2,                # Maximum hands to detect
    min_detection_confidence=0.5,   # Minimum confidence to start tracking
    min_tracking_confidence=0.5     # Minimum confidence to keep tracking
)

# 2. Start Video Capture (0 is usually the default laptop webcam)
cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 3. Flip the image horizontally for a later selfie-view display
    # and convert the BGR image to RGB (MediaPipe requires RGB)
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 4. Process the image and find hands
    results = hands.process(image_rgb)

    # 5. Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the 21 landmarks and the connections between them
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # 6. Display the output
    cv2.imshow('Hand Scanner', image)

    # Quit if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 7. Clean up
cap.release()
cv2.destroyAllWindows()