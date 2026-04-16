import cv2
import time
import random
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import winsound

# Constants
PERSON_CLASS = 0
ANIMAL_CLASSES = [15, 16, 17, 18, 19]     # cat, dog, horse, sheep, cow
PLANT_CLASSES = [58, 50]                  # potted plant, broccoli
TELEMETRY_UPDATE_INTERVAL = 5.0

# Alphabetical PlantVillage Classes (for missing model configs)
PLANT_VILLAGE_CLASSES = [
    'Apple Black Rot', 'Apple Cedar Rust', 'Apple Healthy', 'Apple Scab', 'Blueberry Healthy', 
    'Cherry Healthy', 'Cherry Powdery Mildew', 'Corn Cercospora Leaf Spot', 'Corn Common Rust', 
    'Corn Healthy', 'Corn Northern Leaf Blight', 'Grape Black Rot', 'Grape Esca', 'Grape Healthy', 
    'Grape Leaf Blight', 'Orange Huanglongbing', 'Peach Bacterial Spot', 'Peach Healthy', 
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy', 'Potato Early Blight', 'Potato Healthy', 
    'Potato Late Blight', 'Raspberry Healthy', 'Soybean Healthy', 'Squash Powdery Mildew', 
    'Strawberry Healthy', 'Strawberry Leaf Scorch', 'Tomato Bacterial Spot', 'Tomato Early Blight', 
    'Tomato Healthy', 'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Mosaic Virus', 
    'Tomato Septoria Leaf Spot', 'Tomato Spider Mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus'
]

# Granular Thresholds
CONF_PERSON = 0.25
CONF_ANIMAL = 0.55  # Higher threshold for animals to reduce noise
CONF_PLANT = 0.18   # Lowered significantly for images on mobile screens

class AgriTelemetry:
    def __init__(self):
        self.temperature = 25.0
        self.humidity = 50.0
        self.soil_moisture = 45.0
        self.last_update = time.time()
        self.status = "Optimal"

    def update_values(self):
        # Random walk for smooth transitions
        delta_temp = random.uniform(-0.8, 0.8)
        delta_hum = random.uniform(-2.0, 2.0)
        delta_soil = random.uniform(-1.5, 1.5)

        self.temperature = max(22.0, min(35.0, round(self.temperature + delta_temp, 1)))
        self.humidity = max(40.0, min(75.0, round(self.humidity + delta_hum, 1)))
        self.soil_moisture = max(30.0, min(60.0, round(self.soil_moisture + delta_soil, 1)))
        
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

class PlantHealthClassifier:
    """Uses a Hugging Face pipeline for true plant disease classification."""
    def __init__(self):
        print("[INFO] Loading Hugging Face Plant Disease Model (Vision Transformer)...")
        try:
            # Connects to HuggingFace Hub to download a specialized PlantVillage model
            self.classifier = pipeline("image-classification", model="ahmed792002/vit-plant-classification")
        except Exception as e:
            print(f"[ERROR] Could not load HF model: {e}. Falling back to mock data.")
            self.classifier = None

    def classify(self, roi):
        if self.classifier is None:
            states = ["Healthy", "Healthy", "Healthy", "Leaf Rust", "Powdery Mildew"]
            return random.choice(states)
            
        try:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_roi)
            
            results = self.classifier(pil_image)
            best_label = results[0]['label']
            
            # Map generic 'LABEL_X' back to standard PlantVillage classes if necessary
            if best_label.startswith('LABEL_'):
                try:
                    idx = int(best_label.replace('LABEL_', ''))
                    clean_label = PLANT_VILLAGE_CLASSES[idx]
                except (ValueError, IndexError):
                    clean_label = best_label
            else:
                clean_label = best_label.replace("___", " - ").replace("_", " ").title()
            
            if "Healthy" in clean_label or "Background" in clean_label:
                return "Healthy"
            # Extract just the disease part if it's too long, but we'll show the whole thing
            return clean_label.split(" - ")[-1] if " - " in clean_label else clean_label
        except Exception as e:
            print(f"[ERROR] Classification failed: {e}")
            return "Unknown"

class SmartAgriLauncher:
    def __init__(self):
        self.root = TkinterDnD.Tk()
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
        self.plant_menu = tk.Toplevel(self.root)
        self.plant_menu.title("Plant Input Source")
        self.plant_menu.geometry("350x250")
        self.plant_menu.configure(bg=self.root.cget('bg'))
        
        lbl = tk.Label(self.plant_menu, text="Choose Plant Image Source", font=("Helvetica", 12, "bold"), bg=self.root.cget('bg'), fg=self.title_label.cget('fg'))
        lbl.pack(pady=15)
        
        btn_cam = tk.Button(self.plant_menu, text="Use Webcam", command=lambda: self.start_plant_mode(0), width=25, bg=self.btn_plant.cget('bg'), fg=self.btn_plant.cget('fg'))
        btn_cam.pack(pady=5)
        
        btn_up = tk.Button(self.plant_menu, text="Upload Photo", command=self.upload_photo, width=25, bg=self.btn_plant.cget('bg'), fg=self.btn_plant.cget('fg'))
        btn_up.pack(pady=5)
        
        dnd_lbl = tk.Label(self.plant_menu, text="(Or Drag & Drop an Image File Here)", font=("Helvetica", 9), bg=self.root.cget('bg'), fg=self.title_label.cget('fg'))
        dnd_lbl.pack(pady=15)
        
        self.plant_menu.drop_target_register(DND_FILES)
        self.plant_menu.dnd_bind('<<Drop>>', self.on_drop_image)
        
    def upload_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        if file_path:
            self.start_plant_mode(file_path)
            
    def on_drop_image(self, event):
        file_path = event.data
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            self.start_plant_mode(file_path)
        else:
            messagebox.showerror("Error", "Please drop a valid image file.")

    def start_plant_mode(self, source):
        if hasattr(self, 'plant_menu') and self.plant_menu.winfo_exists():
            self.plant_menu.destroy()
        self.root.withdraw()
        self.run_plant_module(source)
        self.root.deiconify()

    def play_alert_sound(self):
        """Plays a non-blocking beep alert."""
        def _beep():
            try:
                winsound.Beep(1000, 250)
            except:
                pass
        threading.Thread(target=_beep, daemon=True).start()

    def run_intrusion_module(self):
        print("[INFO] Loading Intrusion Model...")
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(0)
        
        # Stability filters for false positive reduction
        animal_hits = 0
        last_beep_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # We use a lower internal threshold for prediction then filter manually
            results = model.predict(source=frame, stream=True, verbose=False, conf=0.15)
            intrusion_detected = False
            cur_frame_animal = False

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    is_person = (cls == PERSON_CLASS and conf > CONF_PERSON)
                    is_animal = (cls in ANIMAL_CLASSES and conf > CONF_ANIMAL)

                    if is_person or is_animal:
                        if is_animal:
                            cur_frame_animal = True
                            # Only alert on animals after 3 frames of stable detection
                            if animal_hits < 3:
                                continue
                        
                        intrusion_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = (0, 0, 255) if is_person else (255, 165, 0) # Red for person, Orange for animal
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ALERT: {model.names[cls]} {conf:.2f}", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update animal stability counter
            if cur_frame_animal:
                animal_hits = min(animal_hits + 1, 10)
            else:
                animal_hits = max(animal_hits - 1, 0)

            if intrusion_detected:
                # Trigger beep alert (max once per second)
                if time.time() - last_beep_time > 1.0:
                    self.play_alert_sound()
                    last_beep_time = time.time()

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

    def resize_for_display(self, image, max_width=1280, max_height=720):
        h, w = image.shape[:2]
        if w > max_width or h > max_height:
            scale = min(max_width/w, max_height/h)
            return cv2.resize(image, (int(w*scale), int(h*scale)))
        return image

    def run_plant_module(self, source=0):
        print(f"[INFO] Loading Plant Monitoring Models... Source: {source}")
        yolo_plant = YOLO("yolov8n.pt")
        health_classifier = PlantHealthClassifier()
        telemetry = AgriTelemetry()
        
        is_image = isinstance(source, str)
        if is_image:
            frame = cv2.imread(source)
            if frame is None:
                messagebox.showerror("Error", "Could not read image file.")
                return
            frame = self.resize_for_display(frame)
            
            # Predict once for image
            results = yolo_plant.predict(source=frame, stream=False, verbose=False, conf=CONF_PLANT)
            cached_boxes = []
            plants_found = 0
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in PLANT_CLASSES:
                        plants_found += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        roi = frame[y1:y2, x1:x2]
                        health_state = health_classifier.classify(roi)
                        cached_boxes.append((x1, y1, x2, y2, health_state))
            cap = None
        else:
            cap = cv2.VideoCapture(source)
            plants_found = 0
        
        while True:
            if not is_image:
                ret, frame = cap.read()
                if not ret: break
                
                results = yolo_plant.predict(source=frame, stream=True, verbose=False, conf=CONF_PLANT)
                curr_boxes = []
                plants_found = 0
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls in PLANT_CLASSES:
                            plants_found += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            roi = frame[y1:y2, x1:x2]
                            health_state = health_classifier.classify(roi)
                            curr_boxes.append((x1, y1, x2, y2, health_state))
                draw_boxes = curr_boxes
            else:
                draw_boxes = cached_boxes

            display_frame = frame.copy()
            telemetry.check_and_update()
            
            for (x1, y1, x2, y2, health_state) in draw_boxes:
                if health_state == "Healthy":
                    if telemetry.status == "Optimal":
                        color = (0, 255, 0) # Green
                    else:
                        color = (0, 255, 255) # Yellow
                else:
                    color = (0, 0, 255) # Red
                    
                label = f"Crop: {health_state} ({telemetry.status})"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Ensure the text isn't drawn off the top edge of the image
                text_y = max(20, y1 - 10)
                # Ensure x1 isn't negative
                text_x = max(0, x1)
                
                # Add a subtle black background to the text so it's readable on varying backgrounds
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display_frame, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), (0, 0, 0), -1)
                
                cv2.putText(display_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Render Theme-Aware HUD
            txt_color = self.draw_hud_panel(display_frame, 10, 10, 260, 160)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, "PLANT MONITORING STATUS", (20, 35), font, 0.6, txt_color, 2)
            cv2.putText(display_frame, f"Temp: {telemetry.temperature} C", (20, 65), font, 0.5, txt_color, 1)
            cv2.putText(display_frame, f"Humidity: {telemetry.humidity} %", (20, 85), font, 0.5, txt_color, 1)
            cv2.putText(display_frame, f"Soil Moist: {telemetry.soil_moisture} %", (20, 105), font, 0.5, txt_color, 1)
            cv2.putText(display_frame, f"System: {telemetry.status}", (20, 125), font, 0.5, txt_color, 1)
            
            # Additional detection feedback
            status_color = (0, 255, 0) if plants_found > 0 else (0, 165, 255)
            cv2.putText(display_frame, f"Crops in view: {plants_found}", (20, 145), font, 0.5, status_color, 2)

            cv2.putText(display_frame, "Mode: Plant Health | 'm' for Menu", (10, display_frame.shape[0]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Plant Monitoring Mode", display_frame)
            key = cv2.waitKey(200 if is_image else 1) & 0xFF
            if key == ord('m') or key == ord('q'):
                break

        if cap:
            cap.release()
        cv2.destroyAllWindows()
        del yolo_plant # Unload from memory

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    launcher = SmartAgriLauncher()
    launcher.start()
