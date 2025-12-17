import argparse
import logging
import time
import cv2
import mediapipe as mp
import math
import json
import os
import sys

''' Create a venv first and install dependencies '''

# Add lerobot source to path
LEROBOT_SRC_PATH = "/Users/chris/Library/CloudStorage/OneDrive-Personal/Grad/research/robots/so-arm101/lerobot/src"
if os.path.exists(LEROBOT_SRC_PATH) and LEROBOT_SRC_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_SRC_PATH)

# Try to import the LeKiwi client from the `lerobot` package. If the package
# isn't available in PYTHONPATH, the script will still run and only print
# mapped robot actions instead of sending them.
try:
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig

    HAS_LEKIWI = True
except Exception:
    print("No robot")
    LeKiwiClient = None
    LeKiwiClientConfig = None
    HAS_LEKIWI = False
# use the media pipe to teleoperate the robot when using a camera. Not really from the robot itself, but a farmer using a camera to control the robot remotely. This needs the hand tracking to work well with the range of motion of the follower arms.

'''
TODO: Integrate this with motor controls to send commands to the robot based on recognized gestures.
    Use YoLov7 and open pose to compare qualitative performance.
'''


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
    return tip.y < dip.y  # for a camera in front, smaller y -> higher in image


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
    if (
        not thumb_open
        and not index_open
        and not middle_open
        and not ring_open
        and not pinky_open
    ):
        return "Start"

    # 2. Open Hand
    if thumb_open and index_open and middle_open and ring_open and pinky_open:
        return "STOP"

    # 3. Thumbs Up
    if (
        thumb_open
        and not index_open
        and not middle_open
        and not ring_open
        and not pinky_open
    ):
        return "SPIN"

    # 4. Pointing (Index only)
    if index_open and not middle_open and not ring_open and not pinky_open:
        return "LEFT"

    # 5. Peace sign
    if index_open and middle_open and not ring_open and not pinky_open:
        return "RIGHT"

    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description="Hand-gesture driven robot teleop")
    parser.add_argument(
        "--remote-ip", help="LeKiwi host IP (if empty, actions are printed)"
    )
    parser.add_argument("--id", default="camera_teleop", help="Client id")
    parser.add_argument("--fps", type=int, default=30, help="Loop FPS")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    robot = None
    if args.remote_ip:
        if not HAS_LEKIWI:
            logging.warning(
                "LeKiwi client classes not importable. Actions will only be printed."
            )
        else:
            try:
                cfg = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.id)
                robot = LeKiwiClient(cfg)
                robot.connect()
                logging.info("Connected to LeKiwi host %s", args.remote_ip)
            except Exception as e:
                logging.exception(
                    "Failed to connect to LeKiwi host; continuing without robot: %s", e
                )

    # Start video capture
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize MediaPipe Hands
    # static_image_mode=False → video mode
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # could track 2 hands if desired
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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

            # Map gesture to robot action (base velocities by default)
            def gesture_to_action(label: str) -> dict:
                '''
                fist - start
                open hand - stop
                thumbs up - spin
                pointing - left turn
                peace sign - right turn
                '''
                # velocities: x.vel (m/s), y.vel (m/s), theta.vel (deg/s)
                slow = 0.1
                medium = 0.2
                fast = 0.4
                spin = 180.0
                if label == "Start":
                    return {"x.vel": slow, "y.vel": 0.0, "theta.vel": 0.0}
                if label == "STOP":
                    return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                if label == "SPIN":     
                    return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": spin}
                if label == "LEFT":
                    return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 60.0}
                if label == "RIGHT":
                    return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -60.0}
                # Default: do not move
                return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

            # Prepare and send action (only send when changed)
            action = gesture_to_action(gesture_label)
            if not hasattr(main, "_last_sent_action"):
                main._last_sent_action = None
            if action != main._last_sent_action:
                if robot is not None:
                    try:
                        robot.send_action(action)
                        logging.info("Sent action to robot: %s", action)
                    except Exception:
                        logging.exception("Failed to send action to robot")
                else:
                    logging.info(
                        "(no robot) Mapped gesture '%s' -> action: %s",
                        gesture_label,
                        json.dumps(action),
                    )
                main._last_sent_action = action

            # display label on screen
            cv2.putText(
                frame,
                f"Gesture: {gesture_label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3,
            )

            cv2.imshow("Hand Gesture Recognition", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    if "robot" in locals() and robot is not None:
        try:
            robot.disconnect()
            logging.info("Robot disconnected")
        except Exception:
            logging.exception("Error while disconnecting robot")


if __name__ == "__main__":
    main()
