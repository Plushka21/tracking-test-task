# Original code:
# https://pynerds.blogspot.com/2021/09/motion-detection-and-tracking-using.html

import cv2
import numpy as np
import argparse
from iou import IoU


def filter_diffrence(frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
    # Find difference between two frames to detect changes
    diff = cv2.absdiff(frame_1, frame_2)

    # Convert difference frame to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply some blur to smoothen the frame
    diff_blur = cv2.GaussianBlur(diff_gray, (13, 13), 7)

    # # Apply dilation and erosion to the frame
    kernel = np.ones((3, 3), np.uint8)
    dilate_img = cv2.dilate(diff_blur, kernel, iterations=2)
    erode_img = cv2.erode(dilate_img, kernel, iterations=3)

    # Apply thresholding
    _, thresh_bin = cv2.threshold(erode_img, 18, 255, cv2.THRESH_BINARY)
    return thresh_bin


def get_contours(
    filtered_diff: np.ndarray,
    all_contours: list[np.ndarray],
    object_counter: int,
    min_area: int,
) -> tuple[list, int]:
    # Find contours
    new_contours, hierarchy = cv2.findContours(
        filtered_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter detected contours
    for contour in new_contours:
        # If area is smaller than minimum value, we skip this contour
        new_area = cv2.contourArea(contour)
        if new_area < min_area:
            continue

        # Otherwise find its center
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        updated = False
        # Compare the current contour with saved contours
        for i, (exist_con, age, exist_counter) in enumerate(all_contours):
            # Take center of saved contour
            M = cv2.moments(exist_con)
            exist_cX = int(M["m10"] / M["m00"])
            exist_cY = int(M["m01"] / M["m00"])

            # If distance between current contour and one of saved contours
            # is smaller than some threshold, then we assume that contours belong to the same object.
            # Hence, old contour may be ignored, thus its second argument is False
            dist = np.sqrt((exist_cX - cX) ** 2 + (exist_cY - cY) ** 2)
            if dist < 100:
                x1, y1, w1, h1 = cv2.boundingRect(exist_con)
                x2, y2, w2, h2 = cv2.boundingRect(contour)

                box1 = [x1, y1, x1 + w1, y1 + w1]
                box2 = [x2, y2, x2 + w2, y2 + w2]
                iou_score = IoU(box1, box2)
                if iou_score > 0.3:
                    all_contours[i] = [contour, 0, exist_counter]
                    updated = True
                    break
        if not updated:
            object_counter += 1
            all_contours.append([contour, 0, object_counter])

    # Keep only youngest contours
    upd_contours = [upd_cnt for upd_cnt in all_contours if upd_cnt[1] < 3]

    return upd_contours, object_counter


def draw_bboxes(frame: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    frame_copy = frame.copy()
    # Draw each contour
    for i in range(len(contours)):
        # Update age of a contour
        contours[i][1] += 1
        real_cont = contours[i][0]
        # Find parameters of appropriate bounding box
        x, y, w, h = cv2.boundingRect(real_cont)
        # Draw rectangle around detected object
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame_copy,
            f"Object #{contours[i][2]}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            2,
        )

    return frame_copy


def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-v", "--video", default="min_trim.mkv", help="path to the video file"
    )
    ap.add_argument(
        "-a", "--min-area", type=int, default=5000, help="minimum area size"
    )
    args = vars(ap.parse_args())

    min_area = args["min_area"]
    # Capturing video
    capture = cv2.VideoCapture(args["video"])

    object_counter = 1

    # List all contours
    all_contours = []

    while capture.isOpened():
        # Read frame by frame
        _, frame_1 = capture.read()
        _, frame_2 = capture.read()

        filtered_diff = filter_diffrence(frame_1, frame_2)

        all_contours, object_counter = get_contours(
            filtered_diff, all_contours, object_counter, min_area
        )

        frame_display = draw_bboxes(frame_1, all_contours)

        # Show result
        cv2.imshow("Detecting Motion...", frame_display)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
