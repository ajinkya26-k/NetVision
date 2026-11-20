# map_net.py
import numpy as np

class ContextualMAPNet:
    """Novelty Direction 2: Contextual Multi-Attribute Priority Regressor"""
    def __init__(self, v_max=50.0):
        self.v_max = v_max
        self.class_urgency = {
            'person': 1.0, 'car': 0.85, 'bus': 0.85, 'truck': 0.80,
            'motorcycle': 0.75, 'bicycle': 0.70, 'backpack': 0.50
        }
        self.default_urgency = 0.35

    def compute_priority(self, class_name, box, conf, prev_box=None, img_shape=(720, 1280)):
        h_img, w_img = img_shape[:2]
        x1, y1, x2, y2 = box
        norm_area = max(1, (x2 - x1) * (y2 - y1)) / (h_img * w_img)

        # 1. Semantic urgency
        u_class = self.class_urgency.get(class_name, self.default_urgency)
        # 2. Scale vulnerability (smaller = higher vulnerability under DCT)
        v_scale = np.exp(-15.0 * np.sqrt(norm_area))
        # 3. Epistemic confidence curve (focus on ambiguous detections)
        phi_conf = 4.0 * conf * (1.0 - conf)
        # 4. Kinematic motion
        if prev_box is not None:
            c_curr = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            c_prev = np.array([(prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0])
            m_norm = min(1.0, np.linalg.norm(c_curr - c_prev) / self.v_max)
        else:
            m_norm = 0.5

        p_score = 0.35 * u_class + 0.25 * v_scale + 0.15 * phi_conf + 0.25 * m_norm
        return float(np.clip(p_score, 0.0, 1.0))

    def map_to_quality(self, p_score):
        """Maps priority to JPEG Quality factor and Tier."""
        if p_score >= 0.70:
            return 92, "HIGH"
        elif p_score >= 0.40:
            return 75, "MED"
        else:
            return 45, "LOW"