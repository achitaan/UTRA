import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import requests


from send import send_json_to_backend

try:
    from src.data import landmark_names
except ImportError:
    landmark_names = {i: str(i) for i in range(33)}

BACKEND_URL = "http://your-backend-server.com/api/handwash-scores"

class HandMovement:
    def __init__(self, plot: bool = False):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            smooth_segmentation=True
        )
        self.plot = plot
        if self.plot:
            self.__plot_init()

    def __plot_init(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.pose_connections = list(self.mp_holistic.POSE_CONNECTIONS)
        self.lines = []
        for _ in self.pose_connections:
            line, = self.ax.plot([0, 0], [0, 0], [0, 0], c='red')
            self.lines.append(line)
        self.texts = []
        for i in range(33):
            txt = self.ax.text(0, 0, 0, landmark_names.get(i, str(i)), color='black')
            self.texts.append(txt)
        self.scatter = self.ax.scatter([], [], [], c='blue', s=20)
        self.ax.set_title("3D Pose (Current Frame)")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=20, azim=-60)

    def body_segmentation(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(rgb)
        if result.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, result.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )
        if result.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, result.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )
        return frame, result

    def run_segmentation(self, frame):
        frame, result = self.body_segmentation(frame)
        return frame, result

def detect_fluorescent_green_in_hand(frame, hand_bbox, lower_green=None, upper_green=None,
                                     min_area=50, max_area=50000, morph_kernel_size=5):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if lower_green is None:
        lower_green = np.array([25, 20, 150], dtype=np.uint8)
    if upper_green is None:
        upper_green = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    x, y, w, h = hand_bbox
    hand_mask = np.zeros_like(mask)
    hand_mask[y:y+h, x:x+w] = mask_closed[y:y+h, x:x+w]
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_frame = frame.copy()
    bounding_boxes = []
    total_green_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cv2.drawContours(output_frame, [cnt], -1, (0, 0, 255), 2)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            bounding_boxes.append((bx, by, bw, bh))
            cv2.rectangle(output_frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            total_green_area += bw * bh
    return output_frame, hand_mask, bounding_boxes, total_green_area

def get_hand_bboxes(results, frame_shape):
    bboxes = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_vals) * frame_shape[1])
            y_min = int(min(y_vals) * frame_shape[0])
            x_max = int(max(x_vals) * frame_shape[1])
            y_max = int(max(y_vals) * frame_shape[0])
            w, h = x_max - x_min, y_max - y_min
            bboxes.append((x_min, y_min, w, h))
    return bboxes

def main():
    cap = cv2.VideoCapture(0)
    hand_movement_detector = HandMovement(plot=False)
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    prev_left_wrist = None
    prev_right_wrist = None
    cumulative_movement = 0
    movement_frame_count = 0

    cumulative_closeness = 0
    closeness_frame_count = 0

    # For coverage, we now only use the last frame where hands are visible.
    last_coverage_score = None

    session_active = False

    # Parameters (adjust as needed):
    FRAME_MOVEMENT_TARGET = 10.0   # Target average movement (pixels per frame) for a 100% movement score.
    HANDS_CLOSE_THRESHOLD = 150.0  # Distance (in pixels) below which hands are considered "close".

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        # Process the frame with the holistic model.
        frame_holistic, holistic_results = hand_movement_detector.run_segmentation(frame)

        # Check for hand landmarks from holistic.
        left_detected = holistic_results.left_hand_landmarks is not None
        right_detected = holistic_results.right_hand_landmarks is not None

        # If at least one hand is visible, update session metrics.
        if left_detected or right_detected:
            session_active = True

            # --- Movement Calculation (Frame-to-Frame Displacement) ---
            movement_values = []
            if left_detected:
                left_wrist = holistic_results.left_hand_landmarks.landmark[0]
                left_wrist_px = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))
                if prev_left_wrist is not None:
                    left_delta = math.hypot(left_wrist_px[0] - prev_left_wrist[0],
                                            left_wrist_px[1] - prev_left_wrist[1])
                    movement_values.append(left_delta)
                prev_left_wrist = left_wrist_px
                cv2.circle(frame_holistic, left_wrist_px, 5, (255, 0, 0), -1)
            else:
                prev_left_wrist = None

            if right_detected:
                right_wrist = holistic_results.right_hand_landmarks.landmark[0]
                right_wrist_px = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))
                if prev_right_wrist is not None:
                    right_delta = math.hypot(right_wrist_px[0] - prev_right_wrist[0],
                                             right_wrist_px[1] - prev_right_wrist[1])
                    movement_values.append(right_delta)
                prev_right_wrist = right_wrist_px
                cv2.circle(frame_holistic, right_wrist_px, 5, (0, 0, 255), -1)
            else:
                prev_right_wrist = None

            if movement_values:
                frame_movement = sum(movement_values) / len(movement_values)
                cumulative_movement += frame_movement
                movement_frame_count += 1
                cv2.putText(frame_holistic, f"Frame Movement: {frame_movement:.1f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                frame_movement = 0

            # --- Closeness Bonus: When both hands are detected, higher score if hands are close together.
            if left_detected and right_detected and prev_left_wrist is not None and prev_right_wrist is not None:
                hands_distance = math.hypot(prev_left_wrist[0] - prev_right_wrist[0],
                                            prev_left_wrist[1] - prev_right_wrist[1])
                # Closeness score: 100% when distance is 0; 0% when distance >= HANDS_CLOSE_THRESHOLD.
                frame_closeness = max(0, (HANDS_CLOSE_THRESHOLD - hands_distance) / HANDS_CLOSE_THRESHOLD) * 100
                cumulative_closeness += frame_closeness
                closeness_frame_count += 1
                cv2.putText(frame_holistic, f"Hands Dist: {int(hands_distance)}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            else:
                frame_closeness = 0

            # --- Coverage Detection (Only for the last visible frame) ---
            # We detect how much fluorescent green is on the hand (i.e. how much soap residue remains).
            # Invert the score: if little green is detected, then coverage score is high.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_results = hands_detector.process(frame_rgb)
            hand_bboxes = get_hand_bboxes(hands_results, frame.shape)
            if hand_bboxes:
                frame_coverage_scores = []
                for bbox in hand_bboxes:
                    # Run green detection within the hand bounding box.
                    annotated_frame, hand_mask, boxes, green_area = detect_fluorescent_green_in_hand(frame, bbox)
                    hand_area = bbox[2] * bbox[3]
                    if hand_area > 0:
                        # Compute the ratio of green area to hand area.
                        ratio = min(green_area / hand_area, 1.0)
                        # Invert the score: if ratio is 0 (no green), then score is 100.
                        coverage_score_for_bbox = (1 - ratio) * 100
                        frame_coverage_scores.append(coverage_score_for_bbox)
                    # Draw the hand bounding box on the holistic frame.
                    x, y, w, h = bbox
                    cv2.rectangle(frame_holistic, (x, y), (x+w, y+h), (255, 255, 0), 2)
                if frame_coverage_scores:
                    # For this frame, use the average of the detected scores.
                    last_coverage_score = sum(frame_coverage_scores) / len(frame_coverage_scores)
                else:
                    last_coverage_score = None
                cv2.putText(frame_holistic, f"Coverage (Inverted): {int(last_coverage_score) if last_coverage_score is not None else 0}%",
                            (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        else:
            # When no hands are detected, consider the session ended.
            if session_active:
                # Compute average movement over the session.
                if movement_frame_count > 0:
                    avg_movement = cumulative_movement / movement_frame_count
                else:
                    avg_movement = 0
                # Normalize the average movement to a score (0-100).
                movement_score = min(avg_movement / FRAME_MOVEMENT_TARGET, 1.0) * 100

                # Compute average closeness if available.
                if closeness_frame_count > 0:
                    avg_closeness = cumulative_closeness / closeness_frame_count
                else:
                    avg_closeness = 0
                    
                final_coverage_score = last_coverage_score if last_coverage_score is not None else 0

                # Overall score = avg(movement, closeness, final coverage)
                overall_score = (movement_score + avg_closeness + final_coverage_score) / 3

                #print("Session Ended. Total Score over time interval:")
                #print(f"  Movement Score: {movement_score:.1f}%")
                #print(f"  Closeness Score: {avg_closeness:.1f}%")
                #print(f"  Final Coverage Score: {final_coverage_score:.1f}%")
                #print(f"  Overall Score: {overall_score:.1f}%\n")

                data_to_send = {
                    "session_id": 1234,
                    "movement_score": movement_score,
                    "closeness_score": avg_closeness,
                    "coverage_score": final_coverage_score,
                    "overall_score": overall_score
                }

                with open('info.txt', 'a') as f: 
                    f.write(overall_score)
                # try:
                #     response = send_json_to_backend(data_to_send, BACKEND_URL)
                    
                #     if response.status_code == 200:
                #         print("Data posted successfully!", response.json())
                #     else:
                #         print("Failed to post data. Status code:", response.status_code)
                #         print("Response text:", response.text)
                # except requests.exceptions.RequestException as e:
                #     print("Error sending data:", e)

                # Reset session variables.
                session_active = False
                cumulative_movement = 0
                movement_frame_count = 0
                cumulative_closeness = 0
                closeness_frame_count = 0
                last_coverage_score = None
                prev_left_wrist = None
                prev_right_wrist = None

        # Display the annotated frame.
        cv2.imshow("Hand Washing Quality", frame_holistic)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
