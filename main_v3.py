# detects what?

import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
from collections import deque

FRAME_SKIP_YOLO = 3     #YOLO her n karede bir
last_tracked_objects = np.empty((0,5))
last_drawn_frame = None
N_plot = 50             #grafikte saklanacak veri miktarı
area_threshold = 800
motion_threshold = 1.0
#smoothing pencereleri
SMOOTH_WINDOW_XY = 10
SMOOTH_WINDOW_Z  = 15

my_tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.3)
my_model = YOLO("yolov8n.pt")
my_capture = cv2.VideoCapture(0)
my_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
my_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

prev_gray = None

#nesne id (önceki kutu alanı ve hareket geçmişi)
prev_areas = {}
prev_motions = {}           #z için son N kare
prev_xy_motions = {}        #x-y için son N kare
prev_area_changes = {}      #z smoothing
motion_history = {}         #canlı grafik

frame_count = 0
last_results = None
camera_motion = 0  #kamera hareketini saklamak için

def draw_graph_panel(motion_history, width=1280, height=200):
    graph = np.ones((height, width, 3), dtype=np.uint8) * 255
    #panel yükseklikleri (farklıyken çalışmıyo)
    h_each = height // 3
    labels = ["dx", "dy", "area change"]

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 128, 0), (128, 0, 255), (0, 200, 200)
    ]

    for i, axis in enumerate(["x", "y", "z"]):
        y0, y1 = i * h_each, (i + 1) * h_each
        cv2.putText(graph, labels[i], (10, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        for j, (obj_id, data) in enumerate(motion_history.items()):
            values = list(data[axis])
            if len(values) < 2:
                continue
            pts = np.array([
                [int(k / N_plot * width), int(y1 - (val * 5 + h_each / 2))]
                for k, val in enumerate(values)
            ])
            color = colors[j % len(colors)]
            cv2.polylines(graph, [pts], False, color, 2)
            cv2.putText(graph, f"ID {obj_id}", (width - 100, y0 + 20 + j * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return graph

while True:
    my_return, my_frame = my_capture.read()
    if not my_return:
        print("No frame received")
        break

    my_frame = cv2.flip(my_frame, 1)
    frame_left = my_frame.copy()
    frame_count += 1

    #YOLO her n karede bir
    if frame_count % FRAME_SKIP_YOLO == 0 or last_results is None:
        # YOLO'yu çalıştır
        last_results = my_model(my_frame, verbose=False)
        yolo_plot_frame = last_results[0].plot()

        # yeni tespitleri çıkar
        my_detections = []
        if last_results[0].boxes is not None and len(last_results[0].boxes) > 0:
            for box in last_results[0].boxes.xyxy.cpu().numpy():
                xmin, ymin, xmax, ymax = box[:4]
                conf = 1.0
                my_detections.append([xmin, ymin, xmax, ymax, conf])
        my_detections = np.array(my_detections) if len(my_detections) > 0 else np.empty((0, 5))

        # yeni takip sonuçları
        last_tracked_objects = my_tracker.update(my_detections)

        # sol paneli de güncelle → bu frame saklanacak
        frame_left = my_frame.copy()
        if last_tracked_objects.ndim == 1:
            last_tracked_objects = last_tracked_objects.reshape(1, -1)
        if last_tracked_objects.shape[0] == 0:
            last_tracked_objects = np.empty((0, 5))

        for det in last_tracked_objects:
            if len(det) < 5:
                continue
            xmin, ymin, xmax, ymax, obj_id = det.astype(int)
            color = (0, 255, 0)
            cv2.rectangle(frame_left, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame_left, f"ID:{int(obj_id)}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        last_drawn_frame = frame_left.copy()

    else:
        # YOLO çalışmıyorsa, en son çizilmiş frame'i göster
        yolo_plot_frame = last_results[0].plot()
        if last_drawn_frame is not None:
            frame_left = last_drawn_frame.copy()
        else:
            frame_left = my_frame.copy()
    tracked_objects = last_tracked_objects

    #optical flow
    gray = cv2.cvtColor(my_frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mask = np.ones(gray.shape, dtype=np.uint8)
        for det in tracked_objects:
            if len(det) < 5:
                continue
            xmin, ymin, xmax, ymax, obj_id = det.astype(int)
            mask[ymin:ymax, xmin:xmax] = 0

        camera_motion = mag[mask == 1].mean() if np.any(mask == 1) else 0

        for det in tracked_objects:
            if len(det) < 5:
                continue
            xmin, ymin, xmax, ymax, obj_id = det.astype(int)
            w, h = xmax - xmin, ymax - ymin
            if w <= 0 or h <= 0:
                continue

            obj_flow_x = flow[ymin:ymax, xmin:xmax, 0]
            obj_flow_y = flow[ymin:ymax, xmin:xmax, 1]
            obj_mag = mag[ymin:ymax, xmin:xmax].mean() if obj_flow_x.size > 0 else 0

            #z smoothing
            relative_motion = obj_mag - camera_motion
            prev_motions.setdefault(obj_id, []).append(relative_motion)
            if len(prev_motions[obj_id]) > SMOOTH_WINDOW_XY:
                prev_motions[obj_id].pop(0)

            area = w * h
            prev_area_changes.setdefault(obj_id, []).append(area - prev_areas.get(obj_id, area))
            if len(prev_area_changes[obj_id]) > SMOOTH_WINDOW_Z:
                prev_area_changes[obj_id].pop(0)
            mean_area_change = np.mean(prev_area_changes[obj_id])
            prev_areas[obj_id] = area

            #x-y smoothing
            prev_xy_motions.setdefault(obj_id, {'dx': [], 'dy': []})
            mean_dx = obj_flow_x.mean() if obj_flow_x.size > 0 else 0
            mean_dy = obj_flow_y.mean() if obj_flow_y.size > 0 else 0
            prev_xy_motions[obj_id]['dx'].append(mean_dx)
            prev_xy_motions[obj_id]['dy'].append(mean_dy)
            if len(prev_xy_motions[obj_id]['dx']) > SMOOTH_WINDOW_XY:
                prev_xy_motions[obj_id]['dx'].pop(0)
                prev_xy_motions[obj_id]['dy'].pop(0)
            smooth_dx = np.mean(prev_xy_motions[obj_id]['dx'])
            smooth_dy = np.mean(prev_xy_motions[obj_id]['dy'])

            #hareket yönü
            if abs(smooth_dx) < motion_threshold and abs(smooth_dy) < motion_threshold:
                direction_xy = "stationary"
            else:
                if abs(smooth_dx) > abs(smooth_dy):
                    direction_xy = "right" if smooth_dx > 0 else "left"
                else:
                    direction_xy = "down" if smooth_dy > 0 else "up"

            if mean_area_change > area_threshold:
                direction_z = "closing"
            elif mean_area_change < -area_threshold:
                direction_z = "moving off"
            else:
                direction_z = "stationary z"

            motion_history.setdefault(obj_id, {
                'x': deque(maxlen=N_plot),
                'y': deque(maxlen=N_plot),
                'z': deque(maxlen=N_plot)
            })
            motion_history[obj_id]['x'].append(smooth_dx)
            motion_history[obj_id]['y'].append(smooth_dy)
            motion_history[obj_id]['z'].append(mean_area_change)

            #moving objects, id - hareket bilgisi
            status = f"ID:{int(obj_id)} {direction_xy}, {direction_z}"
            color = (0, 0, 255) if ("closing" in direction_z or direction_xy != "stationary") else (0, 255, 0)
            cv2.rectangle(frame_left, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame_left, status, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    prev_gray = gray


    if camera_motion > 2.0:
        cv2.putText(frame_left, "camera is moving", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame_left, "camera is stable", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    left_view  = cv2.resize(frame_left, (640, 512))
    right_view = cv2.resize(yolo_plot_frame, (640, 512))

    #iki panel yan yana
    bottom_row = np.hstack((left_view, right_view))

    #canlı grafik
    graph_panel = draw_graph_panel(motion_history, width=1280, height=200)
    combined = np.vstack((graph_panel, bottom_row))

    cv2.imshow("moving objs, obj classification", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

my_capture.release()
cv2.destroyAllWindows()
