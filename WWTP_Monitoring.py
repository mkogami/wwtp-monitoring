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

# --- パス解決 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class TrackedObject:
    def __init__(self, obj_id, center, area):
        self.obj_id = obj_id
        self.center = center
        self.area = area
        self.last_seen = time.time()

class WWTP_Monitor:
    def __init__(self, params):
        self.params = params
        self.dev_id = params['dev_id']
        self.last_send_time = 0
        self.frame_count = 0
        self.next_id = 0
        self.tracked_objects = []
        
        # --- AIBox用：最も成功率の高いGStreamerパイプライン ---
        # 1920x1080が重い場合は、ここを 1280x720 に落として試してください
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={self.dev_id} ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=True max-buffers=1"
        )
        
        logging.info(f"Connecting to sensor-id: {self.dev_id}")
        # 引数はパイプライン文字列1つだけ（OpenCVの仕様に合わせる）
        self.cap = cv2.VideoCapture(gst_pipeline)

        # 背景差分設定
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=params['history'], varThreshold=params['v_thresh'], detectShadows=True
        )

    def send_seeit_event(self, event_type, message, frame=None):
        payload = {"timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'), "event_type": event_type, "message": message}
        if frame is not None:
            try:
                ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret: payload["image_data"] = base64.b64encode(buf).decode('utf-8')
            except Exception: pass
        try:
            requests.post(self.params['api_url'], json=payload, timeout=1.0)
        except Exception: pass

    def run(self):
        if not self.cap.isOpened():
            logging.error(f"Cannot open camera sensor-id {self.dev_id}. Hardware is busy or disconnected.")
            return

        logging.info("Camera opened. Waiting for frames...")

        while True:
            # フレームの読み込み
            ret, frame = self.cap.read()
            
            if not ret:
                # フレームが取れない場合、少し待って再試行（ここで止まるのを防ぐ）
                logging.warning("Frame drop detected. Retrying...")
                time.sleep(0.1)
                continue

            self.frame_count += 1
            current_time = time.time()
            
            # ウォームアップ中は背景学習のみ
            fgmask = self.fgbg.apply(frame)
            if self.frame_count <= self.params['warmup_frames']:
                if self.frame_count % 10 == 0:
                    logging.info(f"Learning background... {self.frame_count}/{self.params['warmup_frames']}")
                continue

            # 動体検知
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            current_detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.params['min_area']: continue
                
                M = cv2.moments(contour)
                if M['m00'] == 0: continue
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                x, y, w, h = cv2.boundingRect(contour)
                current_detections.append({'center': (cx, cy), 'area': area, 'bbox': (x, y, w, h)})

            # 追跡と描画
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

                x, y, w, h = det['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            self.tracked_objects = [obj for obj in updated_objects if current_time - obj.last_seen < self.params['track_timeout']]
            
            # 表示
            cv2.imshow('AIBox_WWTP_Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    # 実行パラメータ
    params = {
        'dev_id': 1,             # ここを1か0に変えて試す
        'min_area': 1000,
        'warmup_frames': 60,
        'v_thresh': 50,
        'history': 500,
        'track_timeout': 1.0,
        'dist_threshold': 100,
        'api_url': 'http://localhost:8080/api/event',
        'send_interval': 30
    }

    WWTP_Monitor(params).run()