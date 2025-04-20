import threading
from queue import Queue

class StreamCapture:
    def __init__(self, output_dir, camera_index=0, frame_rate=30):
        self.output_dir = output_dir
        self.camera_index = camera_index
        self.frame_rate = frame_rate
        self.capture = None
        self.running = False
        self.frame_queue = Queue()
        self.writer_thread = threading.Thread(target=self._write_frames, daemon=True)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_capture(self):
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise Exception(f"Unable to open camera with index {self.camera_index}")

        self.running = True
        self.writer_thread.start()
        print("Starting video capture...")
        self._capture_frames()

    def _capture_frames(self):
        frame_interval = 1 / self.frame_rate
        while self.running:
            start_time = time.time()
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            timestamp = int(time.time() * 1000)
            self.frame_queue.put((timestamp, frame))

            elapsed_time = time.time() - start_time
            time_to_wait = frame_interval - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    def _write_frames(self):
        while self.running or not self.frame_queue.empty():
            try:
                timestamp, frame = self.frame_queue.get(timeout=1)
                filename = os.path.join(self.output_dir, f"frame_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved frame: {filename}")
            except Exception as e:
                print(f"Error writing frame: {e}")

    def stop_capture(self):
        self.running = False
        self.writer_thread.join()
        if self.capture:
            self.capture.release()
        print("Video capture stopped.")
