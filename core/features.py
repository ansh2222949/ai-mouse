import math


def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def extract_features(lm):
    wrist = lm[0]
    middle = lm[9]
    hand_size = dist(wrist, middle) or 1
    return [
        dist(lm[4], lm[8]) / hand_size,
        dist(lm[8], lm[12]) / hand_size,
        1.0 if lm[8].y < lm[6].y else 0.0,
        1.0 if lm[12].y < lm[10].y else 0.0,
        1.0 if lm[16].y < lm[14].y else 0.0,
        1.0 if lm[20].y < lm[18].y else 0.0,
        dist(lm[4], lm[17]) / hand_size
    ]
