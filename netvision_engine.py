# netvision_engine.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from ultralytics import YOLO

class ContextualMAPNet:
    """Novelty Direction 2: Vulnerability-Aware Priority Regressor"""
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
        
        # 1. Semantic urgency U(c_i)
        u_class = self.class_urgency.get(class_name, self.default_urgency)
        # 2. Scale vulnerability V(A_i) (physical block-DCT cutoff)
        v_scale = np.exp(-15.0 * np.sqrt(norm_area))
        # 3. Epistemic uncertainty Phi(C_i) (focus on ambiguous detections)
        phi_conf = 4.0 * conf * (1.0 - conf)
        # 4. Kinematic motion M_norm
        if prev_box is not None:
            c_curr = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            c_prev = np.array([(prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0])
            m_norm = min(1.0, np.linalg.norm(c_curr - c_prev) / self.v_max)
        else:
            m_norm = 0.5

        p_score = 0.35 * u_class + 0.25 * v_scale + 0.15 * phi_conf + 0.25 * m_norm
        return float(np.clip(p_score, 0.0, 1.0))

    def map_to_quality(self, p_score):
        if p_score >= 0.70:
            return 92, "HIGH"
        elif p_score >= 0.40:
            return 75, "MED"
        else:
            return 45, "LOW"


class NetworkJitterEstimator:
    """RFC 3550 Compliant Inter-Arrival Network Jitter Estimator"""
    def __init__(self):
        self.prev_transit = None
        self.jitter_ms = 0.0

    def update(self, send_time, recv_time):
        transit_time = (recv_time - send_time) * 1000.0  # in ms
        if self.prev_transit is not None:
            d = abs(transit_time - self.prev_transit)
            # RFC 3550 smoothing filter: J(i) = J(i-1) + (|D| - J(i-1)) / 16
            self.jitter_ms += (d - self.jitter_ms) / 16.0
        self.prev_transit = transit_time
        return self.jitter_ms, transit_time


class NetVisionMetrics:
    """Computes Global PSNR vs. Target-Local PSNR and SSIM"""
    @staticmethod
    def calculate_fidelity(orig, recon, boxes):
        gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        gray_recon = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY)
        
        # 1. Global Metrics across entire canvas
        global_psnr = psnr(gray_orig, gray_recon)
        global_ssim = ssim(gray_orig, gray_recon)

        # 2. Target-Local PSNR (Measuring only critical semantic patches)
        local_psnrs = []
        for box in boxes:
            x1, y1, x2, y2 = box
            crop_orig = gray_orig[y1:y2, x1:x2]
            crop_recon = gray_recon[y1:y2, x1:x2]
            if crop_orig.size > 0 and crop_orig.shape == crop_recon.shape:
                local_psnrs.append(psnr(crop_orig, crop_recon))

        target_local_psnr = float(np.mean(local_psnrs)) if local_psnrs else global_psnr
        return global_psnr, global_ssim, target_local_psnr