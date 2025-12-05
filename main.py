import cv2
import mediapipe as mp
import math
# use the media pipe to teleoperate the robot when using a camera. Not really from the robot itself, but a farmer using a camera to control the robot remotely. This needs the hand tracking to work well with the range of motion of the follower arms.

#TODO: Integrate this with motor controls to send commands to the robot based on recognized gestures.

# define constants
CAMERA_INDEX = 0  # 0 = default MacBook webcam

# mediapipe initializations
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def finger_is_open(hand_landmarks, finger_tip_id, finger_dip_id) -> bool:
    """
    Returns True if the finger tip is farther from wrist than dip joint. 
    Works well for detecting extended fingers.

    Args:
        hand_landmarks (_type_): _description_
        finger_tip_id (_type_): _description_
        finger_dip_id (_type_): _description_
    Returns:
        bool: True if finger is open, False otherwise.
    """
    tip = hand_landmarks.landmark[finger_tip_id]
    dip = hand_landmarks.landmark[finger_dip_id]
    return tip.y < dip.y # for a camera in front, smaller y -> higher in image


def classify_gesture(hand_landmarks) -> str:
    """
    Very simple rule-based classifier using MediaPipe hand landmark indices.
    Returns a string label for the gesture.

    Args:
        hand_landmarks (_type_): _description_
    Returns:
        str: gesture label
    """
    # landmark indices for fingertips
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    # DIP joints (for comparison)
    THUMB_IP = 3
    INDEX_DIP = 7
    MIDDLE_DIP = 11
    RING_DIP = 15
    PINKY_DIP = 19

    # Finger states
    index_open = finger_is_open(hand_landmarks, INDEX_TIP, INDEX_DIP)
    middle_open = finger_is_open(hand_landmarks, MIDDLE_TIP, MIDDLE_DIP)
    ring_open = finger_is_open(hand_landmarks, RING_TIP, RING_DIP)
    pinky_open = finger_is_open(hand_landmarks, PINKY_TIP, PINKY_DIP)

    # Thumb rule: open if tip is left or right of thumb IP depending on hand
    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[THUMB_IP]
    thumb_open = abs(thumb_tip.x - thumb_ip.x) > 0.02
    # print(f"Thumb_tip.x: {thumb_tip.x}, Thumb_ip.x: {thumb_ip.x}, diff: {abs(thumb_tip.x - thumb_ip.x)}")

    # Gesture rules ---------------------------------------------

    # 1. Fist
    if not thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "Start"

    # 2. Open Hand
    if thumb_open and index_open and middle_open and ring_open and pinky_open:
        return "STOP"

    # 3. Thumbs Up
    if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "SPIN"

    # 4. Pointing (Index only)
    if index_open and not middle_open and not ring_open and not pinky_open:
        return "LEFT"

    # 5. Peace sign
    if index_open and middle_open and not ring_open and not pinky_open:
        return "RIGHT"

    return "UNKNOWN"


def main():
    # Start video capture
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize MediaPipe Hands
    # static_image_mode=False → video mode
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,    # could track 2 hands if desired
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print("Gesture Recognition Running (press 'q' to quit)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert BGR (OpenCV) → RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = hands.process(rgb_frame)
            gesture_label = "No Hand Detected"

            # Draw landmarks if detected
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # classify gesture
                gesture_label = classify_gesture(hand_landmarks)
                
                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # display label on screen
            cv2.putText(
                frame,
                f'Gesture: {gesture_label}',
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3
            )
            
            cv2.imshow("Hand Gesture Recognition", frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
