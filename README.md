<div align="center">
  <h1>🖱️ AI Mouse</h1>
  <h3>Real-Time Hand Gesture Control System</h3>
  <p>
    <b>Computer Vision • Hybrid Machine Learning • Low Latency</b>
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/ML-Hybrid%20(KNN%2BForest)-orange?style=for-the-badge" alt="Machine Learning">
    <img src="https://img.shields.io/badge/Vision-MediaPipe-green?style=for-the-badge&logo=opencv" alt="Computer Vision">
  </p>
</div>

---

## 🧠 Project Overview

**AI Mouse** is a real-time computer vision research project that explores mouse control using hand gestures captured via a standard webcam.

Unlike typical deep learning approaches, this system combines **MediaPipe** hand tracking with a **Hybrid Machine Learning architecture (KNN + Random Forest)**. This specific design choice ensures ultra-low latency, stability, and the ability to adapt to new gestures on the fly without heavy retraining.

> 🎯 **Focus:** System design, real-time performance, and practical ML decision-making rather than deep learning complexity.

---

## ✨ Key Features

* **🎥 Real-Time Tracking:** Uses MediaPipe for robust hand landmark detection.
* **✋ Gesture Control:** Full mouse navigation including **Move, Scroll, and Click**.
* **🧠 Hybrid ML Engine:**
    * **KNN:** For fast, online incremental adaptation.
    * **Random Forest:** For stabilizing confidence scores.
* **🎯 Smart Execution:** Temporal buffering to reduce jitter and false positives.
* **📦 Modular Architecture:** Clean separation between vision, logic, and execution layers.
* **🖥️ Fully Offline:** No internet connection required.

---

## 🧱 Technical Architecture

The system avoids deep learning to prioritize speed and interpretability.

```text
AI_MOUSE/
│
├── core/
│   ├── config.py        # System sensitivity & configuration
│   ├── features.py      # Hand landmark feature extraction
│   ├── model.py         # Hybrid KNN + Random Forest logic
│   ├── actions.py       # PyAutoGUI execution (Mouse/Click)
│   └── __init__.py
│
├── main.py              # Camera loop & orchestration
├── requirements.txt     # Dependencies
└── README.md            # Documentation

```

### 🧠 Why Hybrid ML? (The Research Angle)

This problem demands **ultra-low latency** and **online learning**. Deep learning models often introduce unnecessary overhead.

1. **KNN (K-Nearest Neighbors):** Allows for instant adaptation to a specific user's hand shape.
2. **Random Forest:** Acts as a stabilizer to filter out noise from the webcam.
3. **Result:** A system that is faster and more responsive than heavy neural networks for this specific task.

---

## 🧪 Gestures & Controls

Gestures are trained **live** during runtime to match the user's specific hand.

| ID | Gesture Name | Action |
| --- | --- | --- |
| **1** | **MOVE** | Cursor follows hand movement |
| **2** | **SCROLL** | Scroll Up / Down |
| **3** | **CLICK** | Left Mouse Click |
| **4** | **IDLE** | No Action (Safety state) |

### ⌨️ Keyboard Controls

| Key | Function |
| --- | --- |
| `1` / `2` / `3` / `4` | **Train** the respective gesture (Hold to capture data) |
| `s` | **Save** trained model data locally |
| `r` | **Reset** / Clear current calibration |
| `ESC` | **Exit** the program |

---

## 🚀 How to Run

### 1️⃣ Prerequisites

* Python 3.10+
* A working Webcam

### 2️⃣ Installation

```bash
# Create Virtual Environment
py -3.10 -m venv .venv
.venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

```

### 3️⃣ Execution

```bash
python main.py

```

---

## ⚠️ Safety & Warning

This project creates a virtual mouse interface. To ensure smooth movement, the failsafe is disabled:
`pyautogui.FAILSAFE = False`

**If the mouse behaves unexpectedly or gets stuck:**

1. Press **`ESC`** immediately to kill the script.
2. Or press **`Alt + Tab`** to switch windows.
3. Or close the **OpenCV window**.

> *Use with caution. This behavior is intentional for experimentation.*

---

## 📌 Notes

* **User Specific:** Trained data (`.pkl`) is specific to your hand and lighting conditions. It is not synced to Git.
* **OS:** Designed and tested on **Windows**.
* **Scope:** This is an experimental research project, not intended for production accessibility tools.

---

## 🔮 Future Improvements

* [ ] Visual overlays for gesture confidence.
* [ ] Dynamic sensitivity tuning via GUI.
* [ ] Comparative latency study against CNN models.

---

<div align="center">
<b>Shared for Educational & Research Purposes</b>
</div>

```

```
