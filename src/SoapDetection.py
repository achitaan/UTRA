import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_fluorescent_green_in_hand(frame, hand_bbox, lower_green=None, upper_green=None, min_area=50, max_area=50000, morph_kernel_size=5):
    """
    Detect fluorescent green spots within the hand bounding box.
    """
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Default green range
    if lower_green is None:
        lower_green = np.array([35, 50, 50], dtype=np.uint8)
    if upper_green is None:
        upper_green = np.array([85, 255, 255], dtype=np.uint8)

    # Apply thresholding to detect green
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological closing to merge small patches
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Extract hand ROI from mask
    x, y, w, h = hand_bbox
    hand_mask = np.zeros_like(mask)
    hand_mask[y:y+h, x:x+w] = mask_closed[y:y+h, x:x+w]

    # Find contours within hand region
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw results
    output_frame = frame.copy()
    bounding_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # Draw contour
            cv2.drawContours(output_frame, [cnt], -1, (0, 0, 255), 2)
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return output_frame, hand_mask, bounding_boxes

def get_hand_bbox(results, frame_shape):
    """
    Get the bounding box around the detected hand.
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame_shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame_shape[0]
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame_shape[1]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame_shape[0]

            # Convert to integer bbox
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            w, h = x_max - x_min, y_max - y_min
            return (x_min, y_min, w, h)  # Return hand bounding box

    return None  # No hand detected

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No frame captured

        # Flip frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Detect hand bounding box
        hand_bbox = get_hand_bbox(results, frame.shape)

        if hand_bbox:
            # Apply green detection within the hand bounding box
            annotated_frame, mask, boxes = detect_fluorescent_green_in_hand(frame, hand_bbox)

            # Draw the hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            annotated_frame = frame  # No hand detected, show original frame

        # Show results
        cv2.imshow("Fluorescent Detection", annotated_frame)

        # Press 'ESC' to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
