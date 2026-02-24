import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO
from RPLCD.i2c import CharLCD

class CameraYOLO:
    def __init__(self, model_path="newest3.pt", line_position=300):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        
        # Performance Settings
        self.inference_size = 320  # Resizes frame for AI speed
        self.max_distance = 240     # Max pixels a fish can move between frames
        
        # LCD Setup
        try:
            self.lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2)
            self.update_lcd("System Ready", "Total: 0")
        except:
            print("LCD not found, running without display.")
            self.lcd = None

        # Tracking & Counting State
        self.total_count = 0
        self.line_position = line_position
        self.next_object_id = 0
        self.tracked_objects = {}  # {id: (cx, cy)}
        self.counted_ids = set()   # IDs that already crossed the line

        thread = threading.Thread(target=self.update_frame, daemon=True)
        thread.start()

    def update_lcd(self, line1="", line2=""):
        """Async LCD update to fix the AttributeError in app.py"""
        if self.lcd:
            def _write():
                try:
                    self.lcd.clear()
                    self.lcd.cursor_pos = (0, 0)
                    self.lcd.write_string(str(line1)[:16])
                    self.lcd.cursor_pos = (1, 0)
                    self.lcd.write_string(str(line2)[:16])
                except:
                    pass
            threading.Thread(target=_write, daemon=True).start()

    def update_frame(self):
        while self.running:
            start_time = time.time()
            ret, img = self.cap.read()
            if not ret:
                continue

            # 1. AI DETECTION (Fast mode with imgsz=320)
            results = self.model.predict(img, imgsz=self.inference_size, conf=0.2, verbose=False)
            
            new_centroids = []
            if results[0].boxes:
                # YOLO automatically scales coordinates back to original image size
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    new_centroids.append((cx, cy))
                    
                    # Draw visual feedback
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2. MANUAL CENTROID TRACKING
            new_tracked_objects = {}
            for (ncx, ncy) in new_centroids:
                matched_id = None
                min_dist = self.max_distance
                
                # Check distance against objects from the PREVIOUS frame
                for obj_id, (ocx, ocy) in self.tracked_objects.items():
                    dist = np.hypot(ncx - ocx, ncy - ocy) 
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = obj_id
                
                if matched_id is not None:
                    # Existing fish: check if it just crossed the line (Left -> Right)
                    prev_cx = self.tracked_objects[matched_id][0]
                    if prev_cx <= self.line_position < ncx and matched_id not in self.counted_ids:
                        self.total_count += 1
                        self.counted_ids.add(matched_id)
                        self.update_lcd("Fish Counted!", f"Total: {self.total_count}")
                    
                    new_tracked_objects[matched_id] = (ncx, ncy)
                else:
                    # New fish: assign a new ID
                    new_tracked_objects[self.next_object_id] = (ncx, ncy)
                    self.next_object_id += 1

            # Update the master list for the next frame
            self.tracked_objects = new_tracked_objects

            # 3. UI OVERLAY
            cv2.line(img, (self.line_position, 0), (self.line_position, img.shape[0]), (255, 0, 0), 2)
            cv2.putText(img, f"Count: {self.total_count}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            with self.lock:
                self.frame = img

            # Maintain a steady loop speed
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.03 - elapsed)) # Aim for ~30 FPS

    def get_frame(self):
        with self.lock:
            if self.frame is None: return None
            _, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes()
