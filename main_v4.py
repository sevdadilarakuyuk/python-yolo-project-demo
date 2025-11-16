# Display classification and ID tracking on the same screen
# handle transition from one side to the other, crossing boundaries
# write results to an Excel file

import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime
from collections import Counter
from ultralytics import YOLO
from sort import Sort

model = YOLO("yolov8n.pt")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
H, W, _ = frame.shape

MID_X = W // 2              # orta çizgi
last_positions = {}         # nesne geçmişi
object_class_history = {}   # id bazında son 5 karede görülen sınıflar
active_events = []
DISPLAY_TIME = 2.5
event_log = []              # excel için olay listesi
prev_gray = None
CAMERA_MOVE_THRESH = 2.0    # arka plan ortalama hareket eşiği

# IoU fonksiyonu (takipte kullanılacak)
def iou(bb1, bb2):
    xx1 = max(bb1[0], bb2[0])
    yy1 = max(bb1[1], bb2[1])
    xx2 = min(bb1[2], bb2[2])
    yy2 = min(bb1[3], bb2[3])
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return inter / (area1 + area2 - inter + 1e-6)

frame_count = 0
SKIP_FRAMES = 3  # fps dostu ilerlemek açısından

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue

    frame = cv2.flip(frame, 1)
    results = model(frame, conf=0.2)

    detections = []
    if len(results) > 0:
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])    #sınıf bilgisi
                detections.append([x1, y1, x2, y2, conf, cls_id])

    detections = np.array(detections)                                               #yolonun ham tahminleri
    tracked_objects = tracker.update(detections[:, :5]) if len(detections) else []  #sorttan id atanmış nesneler

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # IoU ile sınıf eşlemesi
        label = "nesne"
        best_iou, best_cls = 0, -1
        for det in detections:
            dx1, dy1, dx2, dy2, conf, cls_id = det
            iou_val = iou([x1, y1, x2, y2], [dx1, dy1, dx2, dy2])
            if iou_val > best_iou:
                best_iou, best_cls = iou_val, int(cls_id)

        if best_cls >= 0:
            label = model.names[best_cls]

        # ID bazlı sınıf kararlılığı
        if obj_id not in object_class_history:
            object_class_history[obj_id] = [label]
        else:
            object_class_history[obj_id].append(label)
            if len(object_class_history[obj_id]) > 5:
                object_class_history[obj_id].pop(0)
        label = Counter(object_class_history[obj_id]).most_common(1)[0][0]

        # Geçiş kontrolü
        if obj_id in last_positions:
            last_x = last_positions[obj_id]
            if last_x < MID_X and cx >= MID_X:
                msg = f"ID {obj_id} ({label}): soldan saga gecti"
                print(msg)
                active_events.append((msg, time.time()))
                event_log.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "object_id": obj_id,
                    "label": label,
                    "direction": "soldan → sağa"
                })
            elif last_x > MID_X and cx <= MID_X:
                msg = f"ID {obj_id} ({label}): sagdan sola gecti"
                print(msg)
                active_events.append((msg, time.time()))
                event_log.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "object_id": obj_id,
                    "label": label,
                    "direction": "sağdan → sola"
                })

        last_positions[obj_id] = cx

        # Kutu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Orta çizgi
    cv2.line(frame, (MID_X, 0), (MID_X, H), (255, 0, 0), 2)

    # Kamera hareket kontrolü (arkaplandan karar veriliyor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    camera_status = "kamera sabit"
    if prev_gray is not None:
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        for det in detections:
            x1, y1, x2, y2, _, _ = map(int, det)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_masked = mag[mask == 255]
        mean_motion = np.mean(mag_masked) if len(mag_masked) > 0 else 0

        if mean_motion > CAMERA_MOVE_THRESH:
            camera_status = "kamera sabit degil"

    prev_gray = gray.copy()

    cv2.putText(frame, camera_status, (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255) if camera_status != "kamera sabit" else (0, 255, 0), 2)

    # Aktif olayları ekranda göster
    now = time.time()
    new_events = []
    y_offset = 80
    for msg, t in active_events:
        if now - t < DISPLAY_TIME:
            cv2.putText(frame, msg, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            new_events.append((msg, t))
            y_offset += 30
    active_events = new_events

    # Tek ekran (yukarıda frame atlarken ayrı göstermek gereksizdi bu durumda)
    cv2.imshow("hareketli nesneler & kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Excel'e tek seferde yaz
if event_log:
    pd.DataFrame(event_log).to_excel("events.xlsx", index=False)
    print("Tüm olaylar events.xlsx dosyasına kaydedildi.")
else:
    print("Hiç olay algılanmadı, excel dosyası oluşturulmadı.")

