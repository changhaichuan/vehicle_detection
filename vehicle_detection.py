import cv2
import time
from utils import *

thresh = 0.2  # Threshold for object detection
nms_threshold = 0.2  # Threshold for non-maximum suppression
video_path = 'data/video_01.mp4'  # Path to the input video
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Path to the model configuration (*.pbtxt)
weightsPath = 'frozen_inference_graph.pb'  # Path to the frozen inference graph (*.pb)
dnn_input_size = 320  # Input image size for the object detection model

# Instantiate video capture
cap = cv2.VideoCapture(video_path)

# Instantiate the object detection model and set input parameter
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(dnn_input_size, dnn_input_size)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Get the perspective transform matrix
ptm = get_ptm()

this_frame_time = 0
last_frame_time = 0
while True:
    ret, img = cap.read()

    if ret:
        # Object detection
        detections, confidence, boxes = net.detect(img, confThreshold=thresh)

        # Vehicle filter to only keep cars, motorcycles, buses, and trucks.
        vehicle_detections = []
        vehicle_confidence = []
        vehicle_boxes = []
        for i, detection in enumerate(detections):
            if detection == 3 or detection == 4 or detection == 6 or detection == 8:
                vehicle_detections.append(detection)
                vehicle_confidence.append(confidence[i])
                vehicle_boxes.append(boxes[i])

        # Non-maximum suppression to reduce overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(vehicle_boxes, vehicle_confidence, thresh, nms_threshold)

        # Detection results visualisation
        for i in indices:
            vehicle_box = vehicle_boxes[i]
            vehicle_detection = vehicle_detections[i]
            x, y, w, h = vehicle_box[0], vehicle_box[1], vehicle_box[2], vehicle_box[3]
            cv2.rectangle(img, (x, y), (x + w, h + y), color=vehicle_colours[classNames[vehicle_detection-1]],
                          thickness=2)

        # Perspective transform using ptm
        bev = cv2.warpPerspective(img, ptm, (1564, 940))

        # Cropping BEV to reduce black areas
        bev = bev[80:, 290:, :]

        # Calculate output frame rate
        this_frame_time = time.time()
        fps = 1 / (this_frame_time - last_frame_time)
        print('Current frame rate: {:1f}'.format(fps))

        last_frame_time = this_frame_time

        # Output frame
        cv2.imshow("Output", bev)
        cv2.waitKey(1)

    else:
        print('End of video')
        break

