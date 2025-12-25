🖱️ AI Mouse — Real-Time Hand Gesture Control
(Computer Vision & Hybrid Machine Learning — Research Project)

AI Mouse is a real-time computer vision research project that explores mouse control using hand gestures captured via a webcam.

The system combines MediaPipe hand tracking with a hybrid machine learning approach (KNN + Random Forest) to achieve low-latency, stable, and adaptive gesture recognition.

This project focuses on system design, real-time performance, and practical ML decision-making, rather than deep learning or production deployment.

✨ Features

🎥 Real-time hand tracking using a standard webcam

✋ Gesture-based mouse control

Move

Scroll

Click

🧠 Hybrid ML decision system

KNN for fast online adaptation

Random Forest for confidence stabilization

🔁 Incremental training during runtime

🎯 Confidence-based action execution

📦 Modular and clean code architecture

🖥️ Fully offline operation
```text
🏗️ Project Architecture
AI_MOUSE/
├── core/
│   ├── config.py        # System & gesture configuration
│   ├── features.py      # Hand landmark feature extraction
│   ├── model.py         # Hybrid KNN + RandomForest logic
│   ├── actions.py       # Mouse / scroll / click execution
│   └── __init__.py
│
├── main.py              # Real-time camera loop & orchestration
├── requirements.txt
├── README.md
```
🧠 Why Hybrid ML (Not Deep Learning)?

This problem demands:

⚡ Ultra-low latency

🧩 Online / incremental learning

🧠 Interpretability and control

Deep learning models introduce unnecessary latency and complexity for this use case.

Instead, this project uses:

KNN for instant adaptation to new gestures

Random Forest for stabilizing predictions

Temporal buffering to reduce jitter

This design makes the system fast, robust, and practical for real-time interaction.

🧪 Gestures
Key	Gesture	Action
1	MOVE	Mouse movement
2	SCROLL	Scroll up / down
3	CLICK	Mouse click
4	IDLE	No action

Gestures are trained live during runtime and are user-specific.

▶️ How to Run
1️⃣ Create & activate virtual environment
```text
py -3.10 -m venv .venv
.venv\Scripts\activate
```
2️⃣ Install dependencies
```text
pip install -r requirements.txt
```
3️⃣ Run the system
```text
python main.py
```
⌨️ Controls
```text
Key	Action
1 / 2 / 3 / 4	Train gesture
s	Save trained data
r	Reset calibration
ESC	Exit program
⚠️ Safety Warning
```
This project disables PyAutoGUI failsafe for smoother control:

pyautogui.FAILSAFE = False


⚠️ If the mouse behaves unexpectedly:
```text
Press ESC

Or Alt + Tab
```
Or close the OpenCV window immediately

Use with caution. This behavior is intentional for experimentation.

📌 Notes

Trained gesture data (.pkl) is user-specific and intentionally ignored in GitHub

Webcam is required

Designed and tested on Windows

Not intended for accessibility or production use

🚀 Future Improvements

Gesture visualization overlays

Dynamic sensitivity tuning

GUI-based training interface

Comparative study with deep learning models

🧠 Key Takeaway

AI Mouse demonstrates that classical machine learning combined with computer vision and careful system design can outperform deep learning for real-time, interactive human–computer interfaces.

📜 License

This project is shared for educational and experimental purposes only.
