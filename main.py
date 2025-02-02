import cv2
from src.HandMovement import HandMovement

def main():
    cap = cv2.VideoCapture(0)
    segmenter = HandMovement(True)

    # Adjust resolution if you like
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, _ = segmenter.run_segmentation(frame)

        cv2.imshow("Holistic + 3D Pose (No YOLO)", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()