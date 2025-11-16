# detects objects

import cv2
from ultralytics import YOLO

my_model = YOLO("yolov8n.pt")
my_capture = cv2.VideoCapture(0)

while True:
    my_return, my_frame = my_capture.read()
    if not my_return:
        break
    my_frame = cv2.flip(my_frame, 1)
    my_annotated_frame = my_model(my_frame, conf=0.5)[0].plot()
    cv2.imshow("Object Detection", my_annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

my_capture.release()
cv2.destroyAllWindows()