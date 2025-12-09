# 🛡️ Real-Time AI Proctoring System  
A lightweight, real-time proctoring system designed to detect cheating behavior during online examinations using computer vision and speech analysis.

This project combines **YOLOv8**, **MediaPipe**, **OpenCV**, and **client-side speech recognition** to monitor participants through their webcam and automatically log suspicious activities.

---

## 📌 Key Features

### 🎯 1. Object Detection (YOLOv8 Nano)
Detects cheating-related objects in real time:
- Mobile phones  
- Books / printed materials  
- Additional persons  

Alerts are triggered when an object is detected for more than a configurable duration.

---

### 👤 2. Face & Gaze Tracking (MediaPipe)
- Detects presence of a face  
- Detects multiple faces  
- Tracks gaze direction (forward, left, right)  
- Alerts when:
  - Face missing for >2 seconds  
  - Looking away for >3 seconds  

---

### 🎙️ 3. Forbidden Speech Detection
Client-side speech recognition listens for suspicious keywords such as:

"tolong, jawab, bocor, jawaban, cheat, bantu, google"


If detected, the system:
- Logs the event  
- Saves a screenshot  
- Stores transcript + forbidden words  

---

### 📁 4. Automatic Evidence Logging
Every violation generates:
- Screenshot (JPEG)  
- JSON metadata  
- Detected objects  
- Face bounding boxes  
- Gaze status  
- Transcript (if speech violation)  

Evidence is stored in:
file path= /evidence/cheat 
file path= /evidence/forbidden


---

### 🖥️ 5. Web Dashboard (viewer.html)
A clean, responsive dashboard for exam supervisors:

- Live camera feed  
- YOLO bounding boxes  
- Face detection overlay  
- Real-time alerts  
- Transcript viewer  
- Evidence log viewer  
- **System ON/OFF toggle**  
- **Evidence folder viewer**  

---

### 🔄 6. System Toggle (ON/OFF)
A single button allows supervisors to pause or resume all detection modules:

- YOLO  
- Face detection  
- Gaze tracking  
- Forbidden speech  
- Evidence logging  

No server restart required.


---

## 📦 Installation

### ✅ Requirements
Python **3.8+**

### ✅ Install dependencies

```bash
pip install -r requirements.txt



▶️ Running the System
1. Start the backend server
bash
python server.py
2. Open the dashboard
Access via browser:

Code
http://localhost:5000


🧪 Usage Guide
✅ Start monitoring
Open the dashboard → system is ON by default.

✅ Pause monitoring

SYSTEM ON/OFF
✅ View evidence
Lihat File Evidence
✅ Check logs
Scroll to the Log Evidence section.

🛠️ Technologies Used
Python + Flask — backend server

OpenCV — camera access & image processing

MediaPipe — face & gaze detection

YOLOv8 Nano — object detection

Web Speech API — speech recognition

HTML + JavaScript — dashboard UI



✅ Why This Project?
This system provides:

A free, open-source alternative to commercial proctoring tools

A transparent and modifiable codebase

A practical example of combining AI + CV + speech recognition

A lightweight solution that runs on any laptop




🤝 Contributing
Contributions are welcome. You may submit:

Bug reports
Feature requests
Pull requests


👤 Author
zaaidil Medan, Indonesia 2025



