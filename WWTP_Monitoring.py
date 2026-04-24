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
        self.dev_id = params['dev_id'] # configから 2 が入るはず
        self.last_send_time = 0
        self.frame_count = 0
        self.next_id = 0
        self.tracked_objects = []
        
        # --- 【修正】USBカメラ(video2)専用のパイプライン ---
        # nvarguscamerasrc を捨てて、USB用の v4l2src に切り替えました
        gst_pipeline = (
            f"v4l2src device=/dev/video{self.dev_id} ! "
            "video/x-raw, width=1280, height=720, framerate=30/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=True max-buffers=1"
        )
        
        logging.info(f"Connecting to USB Camera at /dev/video{self.dev_id}...")
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=params['history'], varThreshold=params['v_thresh'], detectShadows=True
        )

    def send_seeit_event(self, event_type, message, frame=None):
        payload = {"timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'), "event_type": event_type, "message": message, "device_id": self.dev_id}
        if frame is not None:
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret: payload["image_data"] = base64.b64encode(buffer).decode('utf-8')
            except Exception: pass
        try:
            requests.post(self.params['api_url'], json=payload, timeout=2.0)
            logging.info(f"Cloud Event Sent: {message}")
        except Exception: pass

    def run(self):
        if not self.cap.isOpened():
            logging.error(f"Failed to open /dev/video{self.dev_id}. USB connection error.")
            return

        logging.info("USB Camera Connected. Monitoring logic started.")

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            self.frame_count += 1
            current_time = time.time()
            
            fgmask = self.fgbg.apply(frame)
            if self.frame_count <= self.params['warmup_frames']:
                if self.frame_count % 20 == 0:
                    logging.info(f"Warming up... {self.frame_count}/{self.params['warmup_frames']}")
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

            # 追跡
            updated_objects = []
            for det in current_detections:
                matched = next((o for o in self.tracked_objects if math.hypot(det['center'][0]-o.center[0], det['center'][1]-o.center[1]) < self.params['dist_threshold']), None)
                if matched:
                    matched.center, matched.area, matched.last_seen = det['center'], det['area'], current_time
                    updated_objects.append(matched)
                else:
                    new_obj = TrackedObject(self.next_id, det['center'], det['area'])
                    if current_time - self.last_send_time > self.params['send_interval']:
                        self.send_seeit_event("DETECTION", f"New ID:{new_obj.obj_id}", frame=frame)
                        self.last_send_time = current_time
                    self.next_id += 1
                    updated_objects.append(new_obj)
                
                x, y, w, h = det['bbox']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{getattr(matched or new_obj, 'obj_id')}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.tracked_objects = [obj for obj in updated_objects if current_time - obj.last_seen < self.params['track_timeout']]
            
            cv2.imshow('AIBox Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    conf_path = os.path.abspath(os.path.join(CURRENT_DIR, '../conf/app/WWTP_Monitoring.conf'))
    config = configparser.ConfigParser(); config.read(conf_path)
    
    log_file = config.get('logging', 'log_file', fallback='/mnt/userstorage/log/WWTP_Monitoring.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])

    params = {
        'dev_id': config.getint('camera', 'device_id', fallback=2), # 本物のUSBは2
        'min_area': 1200, 'warmup_frames': 60, 'v_thresh': 50, 'history': 500,
        'track_timeout': 1.0, 'dist_threshold': 100,
        'api_url': config.get('seeit', 'api_url', fallback='http://localhost:8080/api/event'),
        'send_interval': 30
    }

    WWTP_Monitor(params).run()