import cv2
import time
import logging
import numpy as np
import math
import requests
import base64

# --- 1. 接続用パイプライン作成関数 ---
def get_pipelines(dev_id, w=1280, h=720):
    return [
        ("CSI-Standard", (f"nvarguscamerasrc sensor-id={dev_id} ! "
                          f"video/x-raw(memory:NVMM), width={w}, height={h}, format=NV12, framerate=30/1 ! "
                          "nvvidconv ! video/x-raw, format=BGRx ! "
                          "videoconvert ! video/x-raw, format=BGR ! appsink drop=True max-buffers=1")),
        ("V4L2-Direct", (f"v4l2src device=/dev/video{dev_id} ! "
                         f"video/x-raw, width={w}, height={h} ! "
                         "videoconvert ! video/x-raw, format=BGR ! appsink drop=True max-buffers=1")),
        ("Simple-Numeric", dev_id)
    ]

class TrackedObject:
    def __init__(self, obj_id, center, area):
        self.obj_id = obj_id
        self.center = center
        self.area = area
        self.last_seen = time.time()

class WWTP_Monitor:
    def __init__(self, dev_id=1):
        self.dev_id = dev_id
        self.frame_count = 0
        self.next_id = 0
        self.tracked_objects = []
        self.last_send_time = 0
        
        # パラメータ設定
        self.warmup_frames = 60
        self.min_area = 1200
        self.dist_threshold = 100
        self.send_interval = 30
        self.api_url = 'http://localhost:8080/api/event'

        # --- カメラ接続トライアル ---
        self.cap = None
        for name, pipe in get_pipelines(self.dev_id):
            logging.info(f"Trying {name}...")
            temp_cap = cv2.VideoCapture(pipe)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    logging.info(f"Successfully connected via {name}!")
                    self.cap = temp_cap
                    break
            temp_cap.release()

        if not self.cap:
            raise Exception(f"Could not open camera sensor-id {self.dev_id}")

        # 背景差分エンジンの初期化
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    def run(self):
        logging.info("Monitoring started. Waiting for warm-up...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Frame capture failed. Retrying...")
                time.sleep(0.1)
                continue

            self.frame_count += 1
            current_time = time.time()
            
            # 動体検知処理
            fgmask = self.fgbg.apply(frame)
            
            if self.frame_count <= self.warmup_frames:
                if self.frame_count % 20 == 0:
                    logging.info(f"Learning background... {self.frame_count}/{self.warmup_frames}")
                continue

            # ノイズ除去
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            current_detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area: continue
                
                M = cv2.moments(contour)
                if M['m00'] == 0: continue
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                x, y, w, h = cv2.boundingRect(contour)
                current_detections.append({'center': (cx, cy), 'area': area, 'bbox': (x, y, w, h)})

            # オブジェクト追跡と描画
            new_tracked_objects = []
            for det in current_detections:
                matched_obj = None
                for obj in self.tracked_objects:
                    dist = math.hypot(det['center'][0] - obj.center[0], det['center'][1] - obj.center[1])
                    if dist < self.dist_threshold:
                        matched_obj = obj
                        break
                
                if matched_obj:
                    matched_obj.center, matched_obj.area, matched_obj.last_seen = det['center'], det['area'], current_time
                    new_tracked_objects.append(matched_obj)
                else:
                    new_obj = TrackedObject(self.next_id, det['center'], det['area'])
                    self.next_id += 1
                    new_tracked_objects.append(new_obj)
                    logging.info(f"New object detected: ID {new_obj.obj_id}")

                # 描画
                x, y, w, h = det['bbox']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, det['center'], 5, (0, 0, 255), -1)

            self.tracked_objects = [obj for obj in new_tracked_objects if current_time - obj.last_seen < 1.0]
            
            # 画面表示
            cv2.imshow('WWTP_AI_Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    # dev_id=1 でダメなら 0 に書き換えてください
    try:
        monitor = WWTP_Monitor(dev_id=1)
        monitor.run()
    except Exception as e:
        logging.error(e)