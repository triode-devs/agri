import cv2
import time
import random
import threading
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO

# Constants
INTRUDER_CLASSES = [0, 16, 17, 18, 19]  # person, dog, horse, sheep, cow
PLANT_CLASS = 58                         # potted plant
CONFIDENCE_THRESHOLD = 0.50
TELEMETRY_UPDATE_INTERVAL = 3.0

class AgriTelemetry:
    def __init__(self):
        self.temperature = 25.0
        self.humidity = 50.0
        self.soil_moisture = 45.0
        self.last_update = time.time()
        self.status = "Optimal"

    def update_values(self):
        self.temperature = round(random.uniform(22.0, 35.0), 1)
        self.humidity = round(random.uniform(40.0, 75.0), 1)
        self.soil_moisture = round(random.uniform(30.0, 60.0), 1)
        
        # Rule Engine
        if self.soil_moisture < 35.0:
            self.status = "Water Required"
        elif self.temperature > 33.0:
            self.status = "Heat Stress"
        else:
            self.status = "Optimal"
            
        self.last_update = time.time()

    def check_and_update(self):
        if time.time() - self.last_update >= TELEMETRY_UPDATE_INTERVAL:
            self.update_values()

class MockPlantHealthClassifier:
    """Simulates a secondary CNN for plant disease classification."""
    def classify(self, roi):
        # In a real app, this would be a CNN inference
        # Here we mock it based on simple randomness for the PoC
        states = ["Healthy", "Healthy", "Healthy", "Leaf Rust", "Powdery Mildew"]
        return random.choice(states)

class SmartAgriLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Agri IoT Framework")
        self.root.geometry("400x500")
        self.is_dark_theme = True
        self.current_model = None
        self.camera = None
        
        # UI Setup
        self.setup_ui()
        self.apply_theme()
        
    def setup_ui(self):
        self.title_label = tk.Label(self.root, text="Smart Agri IoT Framework", font=("Helvetica", 18, "bold"), pady=20)
        self.title_label.pack()

        self.btn_intrusion = tk.Button(self.root, text="Launch Intrusion Detection", 
                                      command=self.launch_intrusion, width=25, height=2, pady=10)
        self.btn_intrusion.pack(pady=10)

        self.btn_plant = tk.Button(self.root, text="Launch Plant Monitoring", 
                                  command=self.launch_plant_monitoring, width=25, height=2, pady=10)
        self.btn_plant.pack(pady=10)

        self.btn_theme = tk.Button(self.root, text="Switch to Light Theme", 
                                  command=self.toggle_theme, width=25)
        self.btn_theme.pack(pady=30)

        self.footer = tk.Label(self.root, text="PoC v1.1 - OpenCV & YOLOv8", font=("Helvetica", 8))
        self.footer.pack(side="bottom", pady=10)

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def apply_theme(self):
        if self.is_dark_theme:
            bg_color = "#141414"
            fg_color = "#FFFFFF"
            btn_bg = "#2D2D2D"
            self.btn_theme.config(text="Switch to Light Theme")
        else:
            bg_color = "#EBEBEB"
            fg_color = "#000000"
            btn_bg = "#DDDDDD"
            self.btn_theme.config(text="Switch to Dark Theme")

        self.root.config(bg=bg_color)
        self.title_label.config(bg=bg_color, fg=fg_color)
        self.footer.config(bg=bg_color, fg=fg_color)
        
        for widget in [self.btn_intrusion, self.btn_plant, self.btn_theme]:
            widget.config(bg=btn_bg, fg=fg_color, activebackground=fg_color, activeforeground=bg_color)

    def draw_hud_panel(self, frame, x, y, w, h):
        """Draws the theme-aware HUD panel."""
        overlay = frame.copy()
        if self.is_dark_theme:
            color = (20, 20, 20)
            text_color = (255, 255, 255)
        else:
            color = (235, 235, 235)
            text_color = (0, 0, 0)
        
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return text_color

    def launch_intrusion(self):
        self.root.withdraw() # Hide menu
        self.run_intrusion_module()
        self.root.deiconify() # Show menu back

    def launch_plant_monitoring(self):
        self.root.withdraw()
        self.run_plant_module()
        self.root.deiconify()

    def run_intrusion_module(self):
        print("[INFO] Loading Intrusion Model...")
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = model.predict(source=frame, stream=True, verbose=False, conf=CONFIDENCE_THRESHOLD)
            intrusion_detected = False

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in INTRUDER_CLASSES:
                        intrusion_detected = True
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"INTRUDER: {model.names[cls]} {conf:.2f}", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if intrusion_detected:
                if int(time.time() * 2) % 2 == 0:
                    cv2.putText(frame, "ALERT: INTRUSION DETECTED", (frame.shape[1]//4, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.putText(frame, "Mode: Intrusion | 'm' for Menu", (10, frame.shape[0]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Intrusion Detection Mode", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m') or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        del model # Unload from memory

    def run_plant_module(self):
        print("[INFO] Loading Plant Monitoring Models...")
        yolo_plant = YOLO("yolov8n.pt")
        health_classifier = MockPlantHealthClassifier()
        telemetry = AgriTelemetry()
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            telemetry.check_and_update()
            results = yolo_plant.predict(source=frame, stream=True, verbose=False, conf=CONFIDENCE_THRESHOLD)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == PLANT_CLASS:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Mock CNN Inference on ROI
                        roi = frame[y1:y2, x1:x2]
                        health_state = health_classifier.classify(roi)
                        
                        # Determine Bounding Box Color
                        # Green: Healthy + Optimal
                        # Yellow: Healthy + Stress
                        # Red: Disease
                        if health_state == "Healthy":
                            if telemetry.status == "Optimal":
                                color = (0, 255, 0) # Green
                            else:
                                color = (0, 255, 255) # Yellow
                        else:
                            color = (0, 0, 255) # Red
                            
                        label = f"Crop: {health_state} ({telemetry.status})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Render Theme-Aware HUD
            txt_color = self.draw_hud_panel(frame, 10, 10, 260, 130)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "PLANT MONITORING STATUS", (20, 35), font, 0.6, txt_color, 2)
            cv2.putText(frame, f"Temp: {telemetry.temperature} C", (20, 65), font, 0.5, txt_color, 1)
            cv2.putText(frame, f"Humidity: {telemetry.humidity} %", (20, 85), font, 0.5, txt_color, 1)
            cv2.putText(frame, f"Soil Moist: {telemetry.soil_moisture} %", (20, 105), font, 0.5, txt_color, 1)
            cv2.putText(frame, f"System: {telemetry.status}", (20, 125), font, 0.5, txt_color, 1)

            cv2.putText(frame, "Mode: Plant Health | 'm' for Menu", (10, frame.shape[0]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Plant Monitoring Mode", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m') or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        del yolo_plant # Unload from memory

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    launcher = SmartAgriLauncher()
    launcher.start()
