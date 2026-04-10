# Smart Agri-IoT Framework Project Plan

## Phase 1: Environment Preparation
Create requirements.txt and main entry point.
- [x] Create `main.py` (with GUI Launcher)
- [x] Create `requirements.txt`
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Ensure `yolov8n.pt` is downloaded

## Phase 2: Core Components Implementation
1. **GUI Launcher (Tkinter)**: 
   - [x] Dark/Light theme toggle logic.
   - [x] Dual-mode selection (Intrusion Detection vs Plant Monitoring).
2. **Module 1: Intrusion Detection**:
   - [x] YOLOv8n inference for intruder classes.
   - [x] High-visibility flashing "ALERT: INTRUSION DETECTED" HUD.
3. **Module 2: Plant Monitoring**:
   - [x] Plant localization (YOLOv8n Class 58).
   - [x] Mock Health Classifier (Secondary CNN simulation).
   - [x] Rule Engine for environmental stress evaluation.
   - [x] Dynamic Bounding Boxes: 
     - **Green**: Healthy/Optimal
     - **Yellow**: Healthy/Stressed
     - **Red**: Disease/Unhealthy

## Phase 3: GUI & Theming
- [x] Persistent theme (Dark/Light) into OpenCV overlay.
- [x] Module isolation (loading/unloading models for memory efficiency).
- [x] 'm' key to safely return to Main Menu.

---

### How to Run Locally

```bash
# 1. Install dependencies
pip install ultralytics opencv-python numpy

# 2. Run the application (this will launch the Main Menu)
python main.py
```

---

### How to Run Locally

```bash
# 1. Install dependencies
pip install ultralytics opencv-python numpy

# 2. Run the application
python main.py
```
