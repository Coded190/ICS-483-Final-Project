import cv2
import time
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
from collections import deque, Counter

gestures = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]
num_frames_per_gesture = 50  # frames per gesture per method
results = {}
SMOOTHING = False         # toggle; set False to disable smoothing
SMOOTH_WINDOW = 5        # majority-vote window in frames
WRIST_CROP_SIZE = 200    # pixels; half-width/half-height around wrist to crop for finger counting

# Resolutions to test: (width, height)
RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]

def classify_gesture_mp(hand_landmarks, handedness):
    """
    Classify gesture from MediaPipe hand landmarks by counting fingers.
    New API: hand_landmarks is a list of NormalizedLandmark objects
    """
    count = 0
    # Indices for fingertip and base (MCP) landmarks
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    mcps = [2, 5, 9, 13, 17]  # Use MCP joints for better accuracy
    
    # Check index to pinky fingers (compare tip to MCP instead of PIP)
    for tip, mcp in zip(tips[1:], mcps[1:]):  # skip thumb
        if hand_landmarks[tip].y < hand_landmarks[mcp].y:
            count += 1
    
    # Thumb: horizontal check for left/right hand (compare tip to MCP)
    thumb_tip = hand_landmarks[tips[0]]
    thumb_mcp = hand_landmarks[mcps[0]]
    
    if handedness == 'Right':
        if thumb_tip.x < thumb_mcp.x:
            count += 1
    else:
        if thumb_tip.x > thumb_mcp.x:
            count += 1
    
    # Map count to gesture name
    if count == 1:      return "ONE"
    elif count == 2:    return "TWO"
    elif count == 3:    return "THREE"
    elif count == 4:    return "FOUR"
    elif count == 5:    return "FIVE"
    return "UNKNOWN"


def classify_gesture_contour(frame):
    """
    Classify gesture using skin color segmentation and convex hull defects.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Skin color range (tunable thresholds) - expanded range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Morphology and blur
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "UNKNOWN"
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 3000:  # Increased threshold
        return "UNKNOWN"
    
    # Approximate contour
    epsilon = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    hull = cv2.convexHull(approx, returnPoints=False)
    if hull is None or len(hull) < 3:
        return "UNKNOWN"
    
    try:
        defects = cv2.convexityDefects(approx, hull)
    except:
        return "UNKNOWN"
        
    if defects is None:
        return "UNKNOWN"
    
    count_defects = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        
        a = np.linalg.norm(np.array(start)-np.array(end))
        b = np.linalg.norm(np.array(start)-np.array(far))
        c = np.linalg.norm(np.array(end)-np.array(far))
        
        # Calculate angle
        if b > 0 and c > 0:
            angle = np.arccos(np.clip((b*b + c*c - a*a) / (2*b*c), -1.0, 1.0))
            # Filter defects by angle and depth
            if angle <= np.pi/2 and d > 10000:  # Added depth threshold
                count_defects += 1
    
    fingers = count_defects + 1
    # Clamp to valid range
    fingers = max(1, min(5, fingers))
    
    if fingers == 1:    return "ONE"
    elif fingers == 2:  return "TWO"
    elif fingers == 3:  return "THREE"
    elif fingers == 4:  return "FOUR"
    elif fingers == 5:  return "FIVE"
    return "UNKNOWN"


# Initialize results dictionary for all methods
for m in ["MediaPipe", "YOLOv8_pose", "OpenCV"]:
    if m not in results:
        results[m] = {"latencies": [], "accuracies": [], "unknowns": [], "fps": [], "resolutions": [], "all_resolutions": {}}

# --- helper smoothing function ---
def smooth_prediction(buffer: deque, new_pred: str, window: int):
    """Push new_pred into buffer and return majority vote when possible."""
    buffer.append(new_pred)
    if len(buffer) < window:
        return Counter(buffer).most_common(1)[0][0]
    vote = Counter(list(buffer)[-window:]).most_common(1)[0][0]
    return vote

# --- YOLOv8-pose block ---
print("Loading YOLOv8 pose model (this may download weights the first time).")
try:
    yolo_pose = YOLO("yolov8n-pose.pt")
    USE_YOLO = True
except Exception as e:
    print("Failed to load YOLOv8-pose model:", e)
    yolo_pose = None
    USE_YOLO = False

yolo_buffer = deque(maxlen=SMOOTH_WINDOW)

# Test each resolution
for resolution in RESOLUTIONS:
    print(f"\n{'='*60}")
    print(f"Testing YOLOv8-pose at resolution: {resolution[0]}x{resolution[1]}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual resolution set: {actual_width}x{actual_height}")
    
    res_key = f"{actual_width}x{actual_height}"
    if res_key not in results["YOLOv8_pose"]["all_resolutions"]:
        results["YOLOv8_pose"]["all_resolutions"][res_key] = {"accuracies": [], "unknowns": [], "fps": [], "latencies": []}
    
    for gesture in gestures:
        correct = 0
        total = 0
        unknown_count = 0
        print(f"  YOLOv8-pose: perform gesture {gesture}")
        
        # Warmup
        for _ in range(5):
            _r, _f = cap.read()
        
        start = time.time()
        for i in range(num_frames_per_gesture):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            t0 = time.time()

            pred = "UNKNOWN"

            if USE_YOLO and yolo_pose is not None:
                try:
                    res_list = yolo_pose.predict(source=frame, conf=0.25, verbose=False)
                    res = res_list[0] if isinstance(res_list, (list, tuple)) else res_list

                    kp_arr = None
                    if hasattr(res, "keypoints") and res.keypoints is not None:
                        try:
                            kp_arr = res.keypoints.xy
                        except Exception:
                            try:
                                kp_arr = np.array(res.keypoints)
                            except Exception:
                                kp_arr = None

                    crop = None
                    if kp_arr is not None and len(kp_arr) > 0:
                        person_kp = kp_arr[0]
                        if person_kp.shape[0] >= 11:
                            lw = person_kp[9]
                            rw = person_kp[10]
                            h, w = frame.shape[:2]
                            
                            try:
                                lx, ly = int(lw[0]), int(lw[1])
                                rx, ry = int(rw[0]), int(rw[1])
                            except Exception:
                                lx = rx = ly = ry = None

                            chosen = None
                            if lx is not None and 0 <= lx < w and 0 <= ly < h:
                                chosen = (lx, ly)
                            elif rx is not None and 0 <= rx < w and 0 <= ry < h:
                                chosen = (rx, ry)

                            if chosen is not None:
                                cx, cy = chosen
                                half = WRIST_CROP_SIZE // 2
                                x0 = max(0, cx - half); y0 = max(0, cy - half)
                                x1 = min(w, cx + half); y1 = min(h, cy + half)
                                crop = frame[y0:y1, x0:x1].copy()
                    
                    if crop is None:
                        if hasattr(res, "boxes") and len(res.boxes.xyxy) > 0:
                            try:
                                x1, y1, x2, y2 = map(int, res.boxes.xyxy[0])
                                bx0 = max(0, x1); by0 = max(0, y1)
                                bx1 = min(frame.shape[1], x2); by1 = min(frame.shape[0], y2)
                                crop = frame[by0:by1, bx0:bx1].copy()
                            except Exception:
                                crop = None

                    if crop is not None and crop.size > 0:
                        pred = classify_gesture_contour(crop)
                    else:
                        pred = "UNKNOWN"
                except Exception as e:
                    if i == 0:
                        print("YOLO pose predict error (will continue):", e)
                    pred = "UNKNOWN"
            else:
                pred = "UNKNOWN"

            if SMOOTHING:
                smoothed = smooth_prediction(yolo_buffer, pred, SMOOTH_WINDOW)
                final_pred = smoothed
            else:
                final_pred = pred

            latency_ms = (time.time() - t0) * 1000
            results["YOLOv8_pose"]["latencies"].append(latency_ms)
            results["YOLOv8_pose"]["all_resolutions"][res_key]["latencies"].append(latency_ms)
            
            if final_pred == gesture:
                correct += 1
            if final_pred == "UNKNOWN":
                unknown_count += 1
            total += 1

        duration = time.time() - start
        avg_fps = total / duration if duration > 0 else 0
        acc = (correct/total * 100) if total>0 else 0
        unk = (unknown_count/total * 100) if total>0 else 0
        
        results["YOLOv8_pose"]["accuracies"].append(acc)
        results["YOLOv8_pose"]["unknowns"].append(unk)
        results["YOLOv8_pose"]["fps"].append(avg_fps)
        results["YOLOv8_pose"]["resolutions"].append(res_key)
        
        results["YOLOv8_pose"]["all_resolutions"][res_key]["accuracies"].append(acc)
        results["YOLOv8_pose"]["all_resolutions"][res_key]["unknowns"].append(unk)
        results["YOLOv8_pose"]["all_resolutions"][res_key]["fps"].append(avg_fps)

    cap.release()


# MediaPipe Hand Tracking
mp_buffer = deque(maxlen=SMOOTH_WINDOW)

import os
import urllib.request
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading MediaPipe hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# Test each resolution
for resolution in RESOLUTIONS:
    print(f"\n{'='*60}")
    print(f"Testing MediaPipe at resolution: {resolution[0]}x{resolution[1]}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual resolution set: {actual_width}x{actual_height}")
    
    res_key = f"{actual_width}x{actual_height}"
    if res_key not in results["MediaPipe"]["all_resolutions"]:
        results["MediaPipe"]["all_resolutions"][res_key] = {"accuracies": [], "unknowns": [], "fps": [], "latencies": []}
    
    for gesture in gestures:
        correct = 0; total = 0; unknown_count = 0
        print(f"  MediaPipe: perform gesture {gesture}")
        
        # Warmup
        for _ in range(5):
            _r,_f = cap.read()
        
        start = time.time()
        for i in range(num_frames_per_gesture):
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            t0 = time.time()
            detection_result = detector.detect(mp_image)
            latency = (time.time() - t0) * 1000
            
            pred = "UNKNOWN"
            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                handedness = detection_result.handedness[0][0].category_name
                pred = classify_gesture_mp(hand_landmarks, handedness)
            
            if SMOOTHING:
                final_pred = smooth_prediction(mp_buffer, pred, SMOOTH_WINDOW)
            else:
                final_pred = pred

            results["MediaPipe"]["latencies"].append(latency)
            results["MediaPipe"]["all_resolutions"][res_key]["latencies"].append(latency)
            
            if final_pred == gesture:
                correct += 1
            if final_pred == "UNKNOWN":
                unknown_count += 1
            total += 1
            
        duration = time.time() - start
        avg_fps = total / duration if duration > 0 else 0
        acc = (correct/total * 100) if total>0 else 0
        unk = (unknown_count/total * 100) if total>0 else 0
        
        results["MediaPipe"]["accuracies"].append(acc)
        results["MediaPipe"]["unknowns"].append(unk)
        results["MediaPipe"]["fps"].append(avg_fps)
        results["MediaPipe"]["resolutions"].append(res_key)
        
        results["MediaPipe"]["all_resolutions"][res_key]["accuracies"].append(acc)
        results["MediaPipe"]["all_resolutions"][res_key]["unknowns"].append(unk)
        results["MediaPipe"]["all_resolutions"][res_key]["fps"].append(avg_fps)

    cap.release()

# ✅ FIXED: Close detector AFTER all resolutions tested
detector.close()


# OpenCV Contour-Based Recognition
ocv_buffer = deque(maxlen=SMOOTH_WINDOW)

for resolution in RESOLUTIONS:
    print(f"\n{'='*60}")
    print(f"Testing OpenCV at resolution: {resolution[0]}x{resolution[1]}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual resolution set: {actual_width}x{actual_height}")
    
    res_key = f"{actual_width}x{actual_height}"
    if res_key not in results["OpenCV"]["all_resolutions"]:
        results["OpenCV"]["all_resolutions"][res_key] = {"accuracies": [], "unknowns": [], "fps": [], "latencies": []}

    for gesture in gestures:
        correct = 0
        total = 0
        unknown_count = 0
        print(f"  OpenCV baseline: perform gesture {gesture}")

        # Warmup
        for _ in range(5):
            _r, _f = cap.read()

        start = time.time()

        for i in range(num_frames_per_gesture):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            t0 = time.time()
            pred = classify_gesture_contour(frame)

            if SMOOTHING:
                final_pred = smooth_prediction(ocv_buffer, pred, SMOOTH_WINDOW)
            else:
                final_pred = pred

            latency = (time.time() - t0) * 1000
            results["OpenCV"]["latencies"].append(latency)
            results["OpenCV"]["all_resolutions"][res_key]["latencies"].append(latency)

            if final_pred == gesture:
                correct += 1
            if final_pred == "UNKNOWN":
                unknown_count += 1

            total += 1

        duration = time.time() - start
        avg_fps = total / duration if duration > 0 else 0
        acc = (correct / total * 100) if total > 0 else 0
        unk = (unknown_count / total * 100) if total > 0 else 0

        results["OpenCV"]["accuracies"].append(acc)
        results["OpenCV"]["unknowns"].append(unk)
        results["OpenCV"]["fps"].append(avg_fps)
        results["OpenCV"]["resolutions"].append(res_key)
        
        results["OpenCV"]["all_resolutions"][res_key]["accuracies"].append(acc)
        results["OpenCV"]["all_resolutions"][res_key]["unknowns"].append(unk)
        results["OpenCV"]["all_resolutions"][res_key]["fps"].append(avg_fps)

    cap.release()


# Save results
with open("results_no_smoothing.json", "w") as f:
    json.dump(results, f, indent=2)

import csv
with open("results_summary_no_smoothing.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["method","resolution","gesture","accuracy(%)","unknown_rate(%)","avg_latency_ms","fps"])
    
    for method in ["MediaPipe", "YOLOv8_pose", "OpenCV"]:
        data = results[method]
        # Use all_resolutions for proper per-gesture tracking
        for res_key, res_data in data["all_resolutions"].items():
            for i, gesture in enumerate(gestures):
                if i < len(res_data["accuracies"]):
                    acc = res_data["accuracies"][i]
                    unk = res_data["unknowns"][i]
                    fps = res_data["fps"][i]
                    avg_lat = sum(res_data["latencies"])/len(res_data["latencies"]) if res_data["latencies"] else 0
                    writer.writerow([method, res_key, gesture, acc, unk, avg_lat, fps])

# Print analysis summary
print("\n" + "="*80)
print("=== OVERALL ANALYSIS OF RESULTS ===")
print("="*80)
for method, data in results.items():
    avg_latency = sum(data["latencies"])/len(data["latencies"]) if data["latencies"] else 0
    fps = (1000/avg_latency) if avg_latency>0 else 0
    avg_acc = sum(data["accuracies"])/len(data["accuracies"]) if data["accuracies"] else 0
    avg_unk = sum(data["unknowns"])/len(data["unknowns"]) if data["unknowns"] else 0
    print(f"\n{method} (Overall):")
    print(f"  Avg Latency = {avg_latency:.1f} ms")
    print(f"  FPS ≈ {fps:.1f}")
    print(f"  Accuracy ≈ {avg_acc:.1f}%")
    print(f"  Unknown Rate ≈ {avg_unk:.1f}%")

# Print per-resolution analysis
print("\n" + "="*80)
print("=== RESOLUTION-SPECIFIC ANALYSIS ===")
print("="*80)
for method, data in results.items():
    print(f"\n{method}:")
    for res_key, res_data in data["all_resolutions"].items():
        if res_data["accuracies"]:
            avg_latency = sum(res_data["latencies"])/len(res_data["latencies"]) if res_data["latencies"] else 0
            fps = (1000/avg_latency) if avg_latency>0 else 0
            avg_acc = sum(res_data["accuracies"])/len(res_data["accuracies"]) if res_data["accuracies"] else 0
            avg_unk = sum(res_data["unknowns"])/len(res_data["unknowns"]) if res_data["unknowns"] else 0
            print(f"  {res_key}:")
            print(f"    Avg Latency = {avg_latency:.1f} ms")
            print(f"    FPS ≈ {fps:.1f}")
            print(f"    Accuracy ≈ {avg_acc:.1f}%")
            print(f"    Unknown Rate ≈ {avg_unk:.1f}%")

print("\n" + "="*80)
print("Benchmark complete! Results saved to:")
print("  - results_no_smoothing.json")
print("  - results_summary_no_smoothing.csv")
print("="*80)