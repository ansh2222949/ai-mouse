import pyautogui
import time
import numpy as np

from .config import (
    SCROLL_SPEED, CLICK_COOLDOWN,
    SWIPE_COOLDOWN, SWIPE_MIN_DIST, SWIPE_MIN_VELOCITY,
    HEIGHT
)

screen_w, screen_h = pyautogui.size()


def execute_action(state, pos, prev_state, box, swipe_hist, last_times):
    x1, y1 = pos
    b_min_x, b_max_x, b_min_y, b_max_y = box
    curr_time = time.time()

    # 1. MOVE
    if state == 1:
        x3 = np.interp(x1, (b_min_x, b_max_x), (0, screen_w))
        y3 = np.interp(y1, (b_min_y, b_max_y), (0, screen_h))
        return {"type": "move", "coords": (x3, y3)}

    # 2. SCROLL
    elif state == 2:
        mid_y = HEIGHT // 2
        if y1 < mid_y - 50:
            pyautogui.scroll(SCROLL_SPEED)
        elif y1 > mid_y + 50:
            pyautogui.scroll(-SCROLL_SPEED)

    # 3. CLICK
    elif state == 3:
        if prev_state != "CLICK" and (curr_time - last_times["click"] > CLICK_COOLDOWN):
            pyautogui.click()
            last_times["click"] = curr_time

    return None
