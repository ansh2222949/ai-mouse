import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from .config import KNN_SLICE_SIZE


class HybridModel:
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.rf = RandomForestClassifier(n_estimators=50)
        self.knn_trained = False
        self.rf_trained = False

    def train_knn(self, X_data, y_data):
        if len(X_data) >= 3:
            self.knn.fit(X_data[-KNN_SLICE_SIZE:], y_data[-KNN_SLICE_SIZE:])
            self.knn_trained = True

    def train_rf(self, X_data, y_data):
        if len(X_data) >= 20 and len(set(y_data)) >= 2:
            self.rf.fit(X_data[-500:], y_data[-500:])
            self.rf_trained = True

    def predict(self, features, X_data, y_data, pred_buffer):
        brain_used = "NONE"
        confidence = 0.0
        final_pred = 4

        if not self.knn_trained:
            return 4, 0.0, "TRAIN MODE"

        # KNN Prediction
        y_knn = y_data[-KNN_SLICE_SIZE:]
        idx = self.knn.kneighbors([features], return_distance=False)[0]
        labels = [y_knn[i] for i in idx]
        knn_pred, knn_votes = Counter(labels).most_common(1)[0]
        knn_conf = knn_votes / 3

        final_pred = knn_pred
        confidence = knn_conf
        brain_used = "KNN"

        # RF Hybrid Logic
        if self.rf_trained:
            rf_probs = self.rf.predict_proba([features])[0]
            rf_conf = np.max(rf_probs)
            rf_pred = np.argmax(rf_probs) + 1

            if knn_conf < 0.6 and rf_conf > 0.75:
                final_pred = rf_pred
                confidence = rf_conf
                brain_used = "RF"
            elif knn_pred == rf_pred:
                confidence = (knn_conf + rf_conf) / 2
                brain_used = "HYBRID"
            else:
                final_pred = pred_buffer[-1] if pred_buffer else 4

        return final_pred, confidence, brain_used
