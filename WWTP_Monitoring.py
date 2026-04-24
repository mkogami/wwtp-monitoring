import cv2
import time
import logging
import numpy as np
import math
import sys

# ターミナルにリアルタイムでログを出す設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_pipelines(dev_id, w=1280, h=720):
    return [
        ("CSI-Standard", (f"nvarguscamerasrc sensor-id={dev_id} ! "
                          f"video/x-raw(memory:NVMM), width={w}, height={h}, format=NV12, framerate=30/1 ! "
                          "nvvidconv ! video/x-raw, format=BGRx ! "
                          "videoconvert ! video/x-raw, format=BGR ! appsink drop=True max-buffers=1")),
        ("V4L2-Direct", (f"v4l2src device=/dev/video{dev_id} ! "
                         f"video/x-raw, width={w}, height={h} ! "
                         "videoconvert ! video/x-raw, format=BGR ! appsink drop=True max-buffers=1")),
        ("Simple-OpenCV", dev_id)
    ]

class WWTP_Monitor:
    def __init__(self, dev_id=1):
        self.dev_id = dev_id
        self.frame_count = 0
        self.cap = None
        
        print("\n=== Camera Connection Phase ===")
        for name, pipe in get_pipelines(self.dev_id):
            logging.info(f"Testing method: [{name}]")
            try:
                self.cap = cv2.VideoCapture(pipe)
                if self.cap.isOpened():
                    # 実際に1枚読めるかテスト
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logging.info(f"✅ SUCCESS: [{name}] is working. Frame size: {frame.shape}")
                        break
                    else:
                        logging.warning(f"❌ FAILED: [{name}] opened but returned empty frame.")
                else:
                    logging.warning(f"❌ FAILED: [{name}] could not be opened.")
                self.cap.release()
                self.cap = None
            except Exception as e:
                logging.error(f"⚠️ ERROR during [{name}]: {e}")

        if self.cap is None:
            print("===============================")
            raise Exception("CRITICAL: All connection methods failed. Is another app using the camera?")

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        print("=== Background Subtractor Initialized ===\n")

    def run(self):
        logging.info("Starting Main Loop. Press 'q' to exit.")
        
        while True:
            t1 = time.time()
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logging.error("Lost frame! Retrying in 0.5s...")
                time.sleep(0.5)
                continue

            self.frame_count += 1
            
            # --- 動体検知 (デバッグ用に検知数も出す) ---
            fgmask = self.fgbg.apply(frame)
            
            if self.frame_count <= 60:
                if self.frame_count % 10 == 0:
                    logging.info(f"Learning background... Frame {self.frame_count}/60")
                continue

            # 輪郭抽出
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            
            detect_count = 0
            for contour in contours:
                if cv2.contourArea(contour) > 1200:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    detect_count += 1

            # フレームごとの処理時間を計測（重くないか確認）
            proc_time = (time.time() - t1) * 1000
            if self.frame_count % 30 == 0:
                logging.info(f"Processing: {proc_time:.1f}ms | Detections: {detect_count}")

            cv2.imshow('WWTP_AI_Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Quit command received.")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("System shutdown gracefully.")

if __name__ == "__main__":
    try:
        # まずは dev_id=1 で試し、ダメなら dev_id=0 を試す
        monitor = WWTP_Monitor(dev_id=1)
        monitor.run()
    except Exception as e:
        logging.error(e)
        print("\n💡 Hint: Try changing 'dev_id=1' to 'dev_id=0' in the code.")