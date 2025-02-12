import cv2
import numpy as np

def detect_fluorescent_green(
    image_path,
    lower_green=None,
    upper_green=None,
    min_area=50,
    max_area=50000,
    morph_kernel_size=5
):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if lower_green is None:
        lower_green = np.array([35, 50, 50], dtype=np.uint8)
    if upper_green is None:
        upper_green = np.array([85, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 4. Morphological closing to merge small bright patches into a continuous region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional: If the spots are very broken up, you can also do a dilation step
    # mask_closed = cv2.dilate(mask_closed, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()
    bounding_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cv2.drawContours(output_image, [cnt], -1, (0, 0, 255), 2)

            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 255), 2)

    return output_image, mask_closed, bounding_boxes

if __name__ == "__main__":
    img_path = r"C:\Programming\UTRA\src\image.png"

    out_img, mask, boxes = detect_fluorescent_green(
        image_path=img_path,
        # Optionally, you can override the ranges:
        # lower_green=np.array([30, 40, 40]),
        # upper_green=np.array([90, 255, 255]),
        morph_kernel_size=5
    )

    print(f"Detected bounding boxes: {boxes}")
    cv2.imshow("Detected Fluorescent Regions", out_img)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
