# detects crossing objects

import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

my_model = YOLO("yolov8n.pt")
my_tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.3) #stabilitesini artırmak için
my_capture = cv2.VideoCapture(0)
my_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
my_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    my_return, my_frame = my_capture.read()
    if not my_return:
        break
    my_frame = cv2.flip(my_frame, 1)
    my_results = my_model(my_frame, conf=0.5, verbose=False)[0]

    my_detections = []
    for box in my_results.boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0]
        conf = float(box.conf[0])
        my_detections.append([xmin.item(), ymin.item(), xmax.item(), ymax.item(), conf])
    if len(my_detections) > 0:
        my_detections = np.array(my_detections, dtype=float).reshape(-1, 5)
    else:
        my_detections = np.empty((0, 5), dtype=float)

    my_tracks = my_tracker.update(my_detections)

    object_positions = {}
    cross_count = 0
    for track in np.atleast_2d(my_tracks):
        if track.shape[0] < 5:
            continue
        xmin, ymin, xmax, ymax, track_id = track.astype(int)
        cv2.rectangle(my_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(my_frame, f"ID {int(track_id)}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        center_x = (xmin + xmax) // 2
        if track_id in object_positions:
            prev_x = object_positions[track_id]
            if prev_x > mid_x and center_x <= mid_x:
                cross_count += 1
            elif prev_x < mid_x and center_x >= mid_x:
                cross_count += 1
        object_positions[track_id] = center_x

    cv2.putText(my_frame, f"Crossed: {cross_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    h, w, _ = my_frame.shape
    mid_x = w // 2
    cv2.line(my_frame, (mid_x, 0), (mid_x, h), (0, 0, 255), 2)

    cv2.imshow("moving objects", my_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

my_capture.release()
cv2.destroyAllWindows()
