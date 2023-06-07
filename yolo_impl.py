import cv2
import numpy as np
import os
import argparse
from iou import IoU


PATH_TO_YOLO = "yolo-coco-model"
CONFIDENCE = 0.7
THRESHOLD = 0.3


class YOLO:
    def __init__(self, PATH_TO_YOLO: str, CONFIDENCE: float, THRESHOLD: float) -> None:
        self.CONFIDENCE = CONFIDENCE
        self.THRESHOLD = THRESHOLD
        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([PATH_TO_YOLO, "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(
            0, 255, size=(len(self.LABELS), 3), dtype="uint8"
        )

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([PATH_TO_YOLO, "yolov3.weights"])
        print(weightsPath)
        configPath = os.path.sep.join([PATH_TO_YOLO, "yolov3.cfg"])

        # load YOLO object detector trained on COCO dataset (80 classes)
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        self.net = net
        self.ln = ln
        self.counter = 0
        # List of boxes that contains: [box, age of box, box number, confidence, classID]
        self.real_boxes = []

    def forward(
        self, frame: np.ndarray
    ) -> tuple[list[tuple[int, int, int, int]], list[float], list[int]]:
        """
        feed an frame to YOLO network, filter weak boxes,
        return boxes, confidence, class labels
        """
        (H, W) = frame.shape[:2]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        new_boxes = []
        new_confidences = []
        new_classIDs = []
        # objects_nums = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions
                if confidence > self.CONFIDENCE:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY + (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    new_boxes.append((x, y, int(width), int(height)))
                    new_confidences.append(float(confidence))
                    new_classIDs.append(classID)

        return new_boxes, new_confidences, new_classIDs

    def filter_boxes(
        self,
        new_boxes: list[tuple[int, int, int, int]],
        new_confidences: list[float],
        new_classIDs: list[int],
    ) -> None:
        # Filter detected objects

        new_confidences = np.array(new_confidences)
        idxs = np.argsort(-new_confidences)
        new_confidences = new_confidences[idxs]
        new_boxes = np.array(new_boxes)[idxs]
        new_classIDs = np.array(new_classIDs)[idxs]

        for j, box in enumerate(new_boxes):
            x, y, w, h = box[0], box[1], box[2], box[3]
            cur_box = (x, y, x + w, y + h)
            cur_conf = new_confidences[j]
            cur_classID = new_classIDs[j]
            updated = False
            for i, (
                exist_box,
                age,
                exist_counter,
                exist_conf,
                exist_classID,
            ) in enumerate(self.real_boxes):
                x, y, w, h = exist_box[0], exist_box[1], exist_box[2], exist_box[3]
                exist_box = (x, y, x + w, y + h)
                # If objects have the same class, compute their IoU
                if cur_classID == exist_classID:
                    iou_score = IoU(cur_box, exist_box)
                    if iou_score > THRESHOLD:
                        self.real_boxes[i] = [
                            box,
                            0,
                            exist_counter,
                            cur_conf,
                            cur_classID,
                        ]
                        updated = True
                        break
            if not updated:
                self.counter += 1
                self.real_boxes.append([box, 0, self.counter, cur_conf, cur_classID])

        self.real_boxes = [box for box in self.real_boxes if box[1] < 3]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        detect objects, supress non maximums, draw boxes
        return frame with boxes
        """
        new_boxes, new_confidences, new_classIDs = self.forward(frame)
        self.filter_boxes(new_boxes, new_confidences, new_classIDs)

        # draw boxes
        for i in range(len(self.real_boxes)):
            cur_object = self.real_boxes[i]
            self.real_boxes[i][1] += 1
            label = self.LABELS[cur_object[4]]
            cur_box = cur_object[0]
            # extract the bounding box coordinates
            (x, y) = (cur_box[0], cur_box[1])
            (w, h) = (cur_box[2], cur_box[3])
            object_num = self.real_boxes[i][2]

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in self.COLORS[cur_object[4]]]
            cv2.rectangle(frame, (x, y), (x + w, y - h), color, 2)
            text = f"#{object_num}, {label}: {cur_object[3]:.4f}"
            cv2.putText(
                frame, text, (x, y - h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        return frame


def main():
    yolo = YOLO(PATH_TO_YOLO, CONFIDENCE, THRESHOLD)
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-v", "--video", default="min_trim.mkv", help="path to the video file"
    )
    args = vars(ap.parse_args())

    # Capturing video
    capture = cv2.VideoCapture(args["video"])

    while capture.isOpened():
        # Read frame by frame
        _, frame = capture.read()

        detected_frame = yolo.detect(frame)
        # Show result
        cv2.imshow("Detecting Motion...", detected_frame)
        if cv2.waitKey(60) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
