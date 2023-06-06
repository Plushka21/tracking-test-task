import cv2
import numpy as np
import os
import argparse
from iou import IoU


PATH_TO_YOLO = "yolo-coco-model"
CONFIDENCE = 0.7
THRESHOLD = 0.6


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
        self.counter = 1

    def forward(self, frame: np.ndarray) -> None:
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
        boxes = []
        confidences = []
        classIDs = []
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
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    # objects_nums.append(self.counter)

        self.boxes = np.array(boxes)
        self.confidences = np.array(confidences)
        self.classIDs = np.array(classIDs)
        # self.objects_nums = np.array(objects_nums)

    def non_max_supression(self) -> None:
        """
        perform non-maximum supression over boxes
        """
        idxs = np.argsort(-self.confidences)
        confidences = self.confidences[idxs]
        boxes = self.boxes[idxs]
        classIDs = self.classIDs[idxs]
        # objects_nums = self.objects_nums[idxs]
        objects_nums = [0] * len(boxes)
        counter = 1

        for i in range(len(boxes)):
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            box1 = (x, y, x + w, y + h)
            for j in range(i + 1, len(boxes)):
                if classIDs[i] == classIDs[j]:
                    # если у них одинаковые классы, вычисляем IoU между ними
                    x, y, w, h = boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]
                    box2 = (x, y, x + w, y + h)
                    if IoU(box1, box2) > self.THRESHOLD:
                        confidences[j] = 0
                        if not confidences[i] and not objects_nums[i]:
                            objects_nums[i] = counter
                            counter += 1
        for i in range(len(objects_nums)):
            if objects_nums[i] == 0:
                objects_nums[i] = counter
                counter += 1
        idxs = np.where(confidences > 0)
        self.boxes = boxes[idxs]
        self.confidences = confidences[idxs]
        self.classIDs = classIDs[idxs]
        self.objects_nums = np.array(objects_nums)[idxs]
        self.counter = len(boxes)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        detect objects, supress non maximums, draw boxes
        return frame with boxes
        """
        self.forward(frame)
        self.non_max_supression()

        # draw boxes
        for i in range(len(self.boxes)):
            label = self.LABELS[self.classIDs[i]]

            # extract the bounding box coordinates
            (x, y) = (self.boxes[i][0], self.boxes[i][1])
            (w, h) = (self.boxes[i][2], self.boxes[i][3])

            object_num = self.objects_nums[i]

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in self.COLORS[self.classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y - h), color, 2)
            text = f"#{object_num}, {label}: {self.confidences[i]:.4f}"
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
