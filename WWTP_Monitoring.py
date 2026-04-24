def __init__(self, params):
        self.params = params
        self.dev_id = params['dev_id']
        self.last_send_time = 0
        self.frame_count = 0
        self.next_id = 0
        self.tracked_objects = []
        
        # 最も安定するV4L2形式（nc_recに近い動き）のパイプラインに変更
        gst_pipeline = (
            f"v4l2src device=/dev/video{self.dev_id} ! "
            "video/x-raw, width=1280, height=720 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=True"
        )
        
        logging.info(f"Connecting to /dev/video{self.dev_id} via V4L2...")
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    def run(self):
        if not self.cap.isOpened():
            logging.error("Failed to open camera.")
            return

        while True:
            # ここで止まっているか確認するためのログ
            # logging.info("Waiting for frame...") 
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logging.warning("Empty frame. Is the camera busy?")
                time.sleep(0.5)
                continue
            
            # 以降、解析処理...