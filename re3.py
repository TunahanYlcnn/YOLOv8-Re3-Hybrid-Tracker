import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tracker.re3_tracker import Re3Tracker

def preprocess(image):
    image = cv2.resize(image, (320, 240))  # RE3 giriş boyutu
    image = image.astype(np.float32) / 255.0  # normalize et
    return image
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / (boxAArea + boxBArea - interArea)


cap = cv2.VideoCapture("C:/Users/tunahan/Desktop/re3Son/dog2.mp4")
model = YOLO("C:/Users/tunahan/Desktop/re3Son/bestv8.pt")
tracker_re3 = Re3Tracker(gpu_id=0, model_path="C:/Users/tunahan/Desktop/re3Son/logs/checkpoints/params.pt")


tracking_initialized = False  # başlangıçta takip başlatılmadı
sayac = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    with torch.no_grad():
        results = model(frame, verbose=False)[0]

    if len(results.boxes.xyxy) > 0:
        def expand_box(box, scale=1.3):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            nw, nh = w * scale, h * scale
            return [cx - nw/2, cy - nh/2, cx + nw/2, cy + nh/2]

        raw_box = results.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, raw_box) 
        yolo_box = expand_box(raw_box, scale=1.3) 

        # Yeşil kutu (YOLO)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # İlk kez takip başlatılacaksa
        if not tracking_initialized:
            rgb_init = frame[:, :, ::-1]
            tracker_re3.track("uav", rgb_init, yolo_box)
            tracking_initialized = True

    def center_distance(boxA, boxB):
        xa, ya = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
        xb, yb = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
        return ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5

    def enhance_contrast(image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    if tracking_initialized:
        rgb = enhance_contrast(frame[:, :, ::-1])

        bbox = tracker_re3.track("uav", rgb)
        print("RE3 bbox:", bbox)

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # RE3: Kırmızı

            if len(results.boxes.xyxy) > 0:
                yolo_box = results.boxes.xyxy[0].cpu().numpy().tolist()
                iou_score = iou(yolo_box, bbox)
                center_dist = center_distance(yolo_box, bbox)

                cv2.putText(frame, f"IoU: {iou_score:.2f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Dist: {center_dist:.1f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if iou_score < 0.02 and center_dist > 50:
                    sayac+=1
                    print("🟠 RE3 takibi başarısız! YOLO kutusuyla yeniden başlatılıyor.",sayac)
                    tracker_re3.reset("uav")
                    tracker_re3.track("uav", rgb, yolo_box)


    cv2.imshow("YOLOv8 + RE3 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()