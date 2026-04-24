import cv2
import configparser
import os
import sys
import time
import logging
import numpy as np
import math
import requests
import base64
import json

# --- パス解決 (DFHC規約準拠) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(CURRENT_DIR, 'lib')
MODEL_DIR = os.path.join(CURRENT_DIR, 'models')
sys.path.append(LIB_DIR)

class TrackedObject:
    def __init__(self, obj_id, center, area, label="unknown"):
        self.obj_id = obj_id
        self.center = center
        self.area = area
        self.label = label
        self.start_time = time.time()
        self.last_seen = time.time()

class WWTP_Monitor:
    def __init__(self, params):
        self.params = params
        self.dev_id = params['dev_id']
        self.last_send_time = 0
        self.frame_count = 0
        self.next_id = 0
        self.tracked_objects = []
        
        # --- ビデオデバイスのオープン (1920x1080 / V4L2) ---
        logging.info(f"Connecting to /dev/video{self.dev_id}")
        self.cap = cv2.VideoCapture(self.dev_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # --- 背景差分エンジンの初期化 ---
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=params['history'], varThreshold=params['v_thresh'], detectShadows=True
        )

        # --- YOLOモデルのロード (OpenCV DNN + CUDA) ---
        model_path = os.path.join(MODEL_DIR, params['model_file'])
        if os.path.exists(model_path):
            logging.info(f"Loading YOLO model: {model_path}")
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.yolo_enabled = True
        else:
            logging.warning("YOLO model not found. Running in Motion-only mode.")
            self.yolo_enabled = False

    def send_seeit_event(self, event_type, message, frame=None):
        payload = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "event_type": event_type,
            "message": message,
            "device_id": self.dev_id
        }
        if frame is not None:
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    payload["image_data"] = base64.b64encode(buffer).decode('utf-8')
                    payload["image_type"] = "image/jpeg"
            except Exception as e:
                logging.warning(f"Image encode error: {e}")

        try:
            requests.post(self.params['api_url'], json=payload, timeout=2.0)
        except Exception:
            pass

    def detect_yolo(self, frame):
        """YOLOによる推論 (GPU加速)"""
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        # ここにバウンディングボックス抽出ロジック(NMS等)を実装
        return outputs

    def run(self):
        if not self.cap.isOpened():
            logging.error(f"Cannot open /dev/video{self.dev_id}.")
            return

        self.send_seeit_event("SYSTEM_START", "AI Monitoring Started.")

        while True:
            ret, frame = self.cap.read()
            if not ret: break

            self.frame_count += 1
            current_time = time.time()
            
            # 1. 背景差分による動体検知 (軽量処理)
            fgmask = self.fgbg.apply(frame)
            if self.frame_count <= self.params['warmup_frames']:
                continue

            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            motion_detected = False
            current_detections = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.params['min_area']: continue
                
                motion_detected = True
                M = cv2.moments(contour)
                if M['m00'] == 0: continue
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                current_detections.append({'center': (cx, cy), 'area': area, 'bbox': cv2.boundingRect(contour)})

            # 2. 動きがあった場合のみYOLOを実行 (GPUリソース節約戦略)
            if motion_detected and self.yolo_enabled and (self.frame_count % 5 == 0):
                # self.detect_yolo(frame) # 必要に応じて詳細解析を実施
                pass

            # 3. トラッキングと描画
            updated_objects = []
            for det in current_detections:
                matched_obj = None
                for obj in self.tracked_objects:
                    dist = math.hypot(det['center'][0] - obj.center[0], det['center'][1] - obj.center[1])
                    if dist < self.params['dist_threshold']:
                        matched_obj = obj
                        break
                
                if matched_obj:
                    matched_obj.center, matched_obj.area, matched_obj.last_seen = det['center'], det['area'], current_time
                    updated_objects.append(matched_obj)
                else:
                    new_obj = TrackedObject(self.next_id, det['center'], det['area'])
                    if current_time - self.last_send_time > self.params['send_interval']:
                        self.send_seeit_event("DETECTION", f"New Object ID:{new_obj.obj_id}", frame=frame)
                        self.last_send_time = current_time
                    self.next_id += 1
                    updated_objects.append(new_obj)

                # 描画 (デバッグ用)
                x, y, w, h = det['bbox']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.tracked_objects = [obj for obj in updated_objects if current_time - obj.last_seen < self.params['track_timeout']]
            
            # 開発環境用表示
            cv2.imshow('DLApp AI Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # 負荷軽減 (AIBox保護)
            time.sleep(0.01)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    conf_path = os.path.abspath(os.path.join(CURRENT_DIR, '../conf/app/WWTP_Monitoring.conf'))
    config = configparser.ConfigParser()
    config.read(conf_path)

    log_file = config.get('logging', 'log_file', fallback='/mnt/userstorage/log/WWTP_Monitoring.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])

    params = {
        'dev_id': config.getint('camera', 'device_id', fallback=1),
        'model_file': config.get('ai', 'model_file', fallback='yolov8n.onnx'),
        'min_area': config.getint('detection', 'min_area', fallback=500),
        'warmup_frames': config.getint('detection', 'warmup_frames', fallback=60),
        'v_thresh': config.getint('detection', 'var_threshold', fallback=50),
        'history': config.getint('detection', 'history', fallback=500),
        'track_timeout': config.getfloat('tracking', 'track_timeout', fallback=1.0),
        'dist_threshold': config.getint('tracking', 'dist_threshold', fallback=150),
        'api_url': config.get('seeit', 'api_url', fallback='http://localhost:8080/api/event'),
        'send_interval': config.getint('seeit', 'send_interval', fallback=30)
    }

    WWTP_Monitor(params).run()