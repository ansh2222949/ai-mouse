import pyautogui

# PyAutoGUI Settings
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Camera / Screen
WIDTH, HEIGHT = 640, 480
SMOOTHING = 5

# Interaction
SCROLL_SPEED = 20
CLICK_COOLDOWN = 0.6
SWIPE_COOLDOWN = 1.0

# ML / Decision
CONFIDENCE_THRESHOLD = 0.6
KNN_SLICE_SIZE = 100
MODEL_FILE = "ai_mouse_data.pkl"

# Physics Engine
SWIPE_MIN_VELOCITY = 1.5
SWIPE_MIN_DIST = 0.3
SWIPE_HISTORY_LENGTH = 10

# Calibration
CALIBRATION_FRAMES = 300

# Gesture Labels
GESTURES = {
    1: "MOVE",
    2: "SCROLL",
    3: "CLICK",
    4: "IDLE"
}
