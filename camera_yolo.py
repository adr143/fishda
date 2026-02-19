import cv2
import threading
import time
from ultralytics import YOLO
from RPLCD.i2c import CharLCD

class CameraYOLO:
    def __init__(self, model_path="yolov8n.pt", line_position=200, target_fps=15):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.lcd = CharLCD(
            i2c_expander='PCF8574',
            address=0x27,
            port=1,
            cols=16,   # change to 20 if using 20x4 LCD
            rows=2,    # change to 4 if using 20x4 LCD
            dotsize=8,
            charmap='A02'
        )

        # Counting
        self.total_count = 0
        self.tracked_objects = {}  # {id: last_x_position}
        self.next_id = 0
        self.line_position = line_position  # x-coordinate of counting line (vertical)

        # FPS
        self.fps = 0
        self.target_fps = target_fps
        self.lcd.cursor_pos = (0,0)
        self.lcd.write_string(f"Total Count:")
        self.lcd.cursor_pos = (1,0)
        self.lcd.write_string(f"{self.total_count}")

        thread = threading.Thread(target=self.update_frame, daemon=True)
        thread.start()

    def update_frame(self):
        while self.running:
            print("Frame Reading")
            start_time = time.time()

            # Prevent lag → grab latest frame
            self.cap.grab()
            ret, img = self.cap.retrieve()
            if not ret:
                continue

            results = self.model(img, stream=True)
            detections = []

            for r in results:
                for box in r.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Center point
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append((cx, cy))

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    # Label (class + confidence)
                    cls = int(box.cls[0]) if box.cls is not None else -1
                    conf = float(box.conf[0]) if box.conf is not None else 0
                    label = f"{self.model.names[cls]} {conf:.2f}" if cls >= 0 else f"Conf {conf:.2f}"

                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            self.update_counts(detections)

            # Draw vertical line
            cv2.line(img, (self.line_position, 0), (self.line_position, img.shape[0]), (0, 255, 0), 2)
            cv2.putText(img, f"Total Fish: {self.total_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # FPS calculation
            end_time = time.time()
            self.fps = 1 / (end_time - start_time + 1e-6)
            cv2.putText(img, f"FPS: {self.fps:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            with self.lock:
                self.frame = img

            # FPS limiter
            elapsed = time.time() - start_time
            sleep_time = max(0, 1/self.target_fps - elapsed)
            time.sleep(sleep_time)

    def update_counts(self, detections):
        new_tracked = {}
        count_changed = False
        for (cx, cy) in detections:
            matched_id = None
            # Match with existing tracked objects
            for obj_id, last_x in self.tracked_objects.items():
                if abs(cx - last_x) < 50:  # near previous x position
                    matched_id = obj_id
                    break

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            # Check crossing the vertical line (left → right)
            if (self.tracked_objects.get(matched_id) is not None
                and self.tracked_objects[matched_id] < self.line_position <= cx):
                self.total_count += 1
                count_changed = True

            new_tracked[matched_id] = cx

        self.tracked_objects = new_tracked
        if count_changed:
            self.lcd.cursor_pos = (0,0)
            self.lcd.write_string(f"Total Count:")
            self.lcd.cursor_pos = (1,0)
            self.lcd.write_string(f"{self.total_count}")

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            _, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes()

    def get_fps(self):
        return self.fps

    def stop(self):
        self.running = False
        self.cap.release()
