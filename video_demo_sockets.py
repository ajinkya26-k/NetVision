# run_video_evaluation.py
import socket
import struct
import pickle
import threading
import time
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from map_net import ContextualMAPNet

HOST = '127.0.0.1'
PORT = 9999
KEYFRAME_INTERVAL = 30

STOP_SIGNAL = threading.Event()
SERVER_READY = threading.Event()

def get_mac_camera():
    """Finds working Mac camera (tries 0, then 1 with AVFoundation)."""
    for idx in [0,1]:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 10:
                print(f"[✓] Using Mac Camera Index: {idx}")
                return cap, False
            cap.release()
    return None, True

# =====================================================================
# SENDER WORKER (Background Thread)
# =====================================================================
def sender_worker():
    SERVER_READY.wait(timeout=5.0)
    time.sleep(0.3)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_sock.connect((HOST, PORT))
    except Exception as e:
        print(f"[SENDER] Socket connect error: {e}")
        STOP_SIGNAL.set()
        return

    model = YOLO('yolov8n.pt')
    map_net = ContextualMAPNet()

    cap, use_synthetic = get_mac_camera()
    if use_synthetic:
        print("[!] Running synthetic animated test stream.")

    track_history = {}
    frame_idx = 0

    print("=== [NETVISION SENDER] Capturing frames and calculating live metrics... ===")
    print("Move around, then sit still to test zero-bandwidth caching. Press 'q' to finish.\n")

    try:
        while not STOP_SIGNAL.is_set():
            if use_synthetic:
                frame = np.full((480, 640, 3), 70, dtype=np.uint8)
                t = time.time()
                cx = int(320 + 160 * np.sin(t * 2))
                cv2.rectangle(frame, (cx, 180), (cx + 80, 300), (0, 0, 220), -1)
                cv2.circle(frame, (cx + 40, 150), 25, (0, 0, 220), -1)
            else:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

            frame_idx += 1
            h, w, c = frame.shape
            is_keyframe = (frame_idx % KEYFRAME_INTERVAL == 1)

            # Baseline Calculation: Standard full-frame JPEG at Q=75 (traditional stream)
            _, std_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            baseline_bytes = len(std_encoded)

            # 1. Run YOLOv8 + ByteTrack
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

            boxes = []
            track_ids = []
            classes = []
            confs = []

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

            payload = {
                'frame_idx': frame_idx,
                'is_keyframe': is_keyframe,
                'shape': (h, w, c),
                'timestamp': time.time(),
                'baseline_bytes': baseline_bytes,
                'bg_data': None,
                'delta_patches': [],
                'static_updates': []
            }

            # 2. I-Frame: 2x downsampled background canvas
            if is_keyframe:
                down_bg = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
                _, bg_enc = cv2.imencode('.jpg', down_bg, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
                payload['bg_data'] = bg_enc.tobytes()

            # 3. P-Frames: Process dynamic targets via MAP-Net
            for box, t_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
                cls_name = model.names[cls_id]
                prev_box = track_history.get(t_id, None)

                p_score = map_net.compute_priority(cls_name, box, conf, prev_box, (h, w))
                q_patch, tier = map_net.map_to_quality(p_score)

                is_moving = True
                if prev_box is not None and not is_keyframe:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    pcx, pcy = (prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2
                    if np.hypot(cx - pcx, cy - pcy) < 3.5:
                        is_moving = False

                if not is_moving:
                    # ZERO-BANDWIDTH CACHE HIT
                    payload['static_updates'].append({
                        'id': t_id,
                        'box': box.tolist(),
                        'p_score': p_score
                    })
                else:
                    x1, y1 = max(0, box[0]), max(0, box[1])
                    x2, y2 = min(w, box[2]), min(h, box[3])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        _, crop_enc = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), q_patch])
                        payload['delta_patches'].append({
                            'id': t_id,
                            'box': box.tolist(),
                            'bytes': crop_enc.tobytes(),
                            'tier': tier,
                            'p_score': p_score
                        })

                track_history[t_id] = box

            # 4. Transmit packet over TCP
            serialized = pickle.dumps(payload)
            header = struct.pack(">I", len(serialized))
            client_sock.sendall(header + serialized)

            time.sleep(0.04)

    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception as e:
        if not STOP_SIGNAL.is_set():
            print(f"[SENDER ERROR]: {e}")
    finally:
        if cap is not None:
            cap.release()
        client_sock.close()


# =====================================================================
# RECEIVER & METRICS ENGINE (Main Thread)
# =====================================================================
def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)

    SERVER_READY.set()
    sender_th = threading.Thread(target=sender_worker, daemon=True)
    sender_th.start()

    conn, addr = server_sock.accept()

    cached_canvas = None
    cached_patches = {}

    # Telemetry Log Buffer
    metrics_log = []

    cv2.namedWindow("NetVision Live Stream", cv2.WINDOW_NORMAL)

    try:
        while not STOP_SIGNAL.is_set():
            header = conn.recv(4)
            if not header or len(header) < 4:
                break
            payload_size = struct.unpack(">I", header)[0]

            data = bytearray()
            while len(data) < payload_size:
                packet = conn.recv(payload_size - len(data))
                if not packet:
                    break
                data.extend(packet)

            recv_time = time.time()
            payload = pickle.loads(data)

            frame_idx = payload['frame_idx']
            h, w, c = payload['shape']
            is_keyframe = payload['is_keyframe']
            send_time = payload['timestamp']
            baseline_bytes = payload['baseline_bytes']
            netvision_bytes = len(data)

            latency_ms = (recv_time - send_time) * 1000.0
            savings_pct = max(0.0, ((baseline_bytes - netvision_bytes) / baseline_bytes) * 100.0)

            # Rebuild base canvas on I-Frames
            if is_keyframe and payload['bg_data'] is not None:
                bg_arr = np.frombuffer(payload['bg_data'], dtype=np.uint8)
                down_bg = cv2.imdecode(bg_arr, cv2.IMREAD_COLOR)
                cached_canvas = cv2.resize(down_bg, (w, h), interpolation=cv2.INTER_LINEAR)

            if cached_canvas is None:
                cached_canvas = np.full((h, w, c), 40, dtype=np.uint8)

            render_frame = cached_canvas.copy()

            # Render 0-byte static objects from cache
            static_count = 0
            for update in payload['static_updates']:
                t_id = update['id']
                x1, y1, x2, y2 = update['box']
                if t_id in cached_patches:
                    crop = cached_patches[t_id]
                    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
                    if crop.shape[1] != cw or crop.shape[0] != ch:
                        crop = cv2.resize(crop, (cw, ch))
                    render_frame[y1:y2, x1:x2] = crop
                    cv2.rectangle(render_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(render_frame, f"ID:{t_id} [0-BYTE STATIC]", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    static_count += 1

            # Render moving delta patches
            delta_count = 0
            for patch in payload['delta_patches']:
                t_id = patch['id']
                x1, y1, x2, y2 = patch['box']
                tier = patch['tier']
                p_score = patch['p_score']

                crop_arr = np.frombuffer(patch['bytes'], dtype=np.uint8)
                crop = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)
                if crop is not None:
                    cached_patches[t_id] = crop
                    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
                    if crop.shape[1] != cw or crop.shape[0] != ch:
                        crop = cv2.resize(crop, (cw, ch))
                    render_frame[y1:y2, x1:x2] = crop

                    color = (0, 255, 0) if tier == "HIGH" else ((0, 165, 255) if tier == "MED" else (0, 0, 255))
                    cv2.rectangle(render_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(render_frame, f"ID:{t_id} P:{p_score:.2f} [{tier}]", 
                                (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    delta_count += 1

            # Log frame metrics for empirical proof
            metrics_log.append({
                'Frame': frame_idx,
                'NetVision_KB': netvision_bytes / 1024.0,
                'Baseline_KB': baseline_bytes / 1024.0,
                'Bandwidth_Saved_Pct': savings_pct,
                'Latency_ms': latency_ms,
                'Moving_Deltas': delta_count,
                'Static_Cache_Hits': static_count,
                'Frame_Type': 'I-FRAME' if is_keyframe else 'P-FRAME'
            })

            # HUD Display
            hud = f"Frame:{frame_idx} | NV:{netvision_bytes/1024:.1f}KB vs STD:{baseline_bytes/1024:.1f}KB (-{savings_pct:.0f}%) | Latency:{latency_ms:.1f}ms"
            cv2.rectangle(render_frame, (0, 0), (w, 35), (20, 20, 20), -1)
            cv2.putText(render_frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow("NetVision Live Stream", render_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                STOP_SIGNAL.set()
                break

    except Exception as e:
        print(f"[RECEIVER ERROR]: {e}")
    finally:
        STOP_SIGNAL.set()
        conn.close()
        server_sock.close()
        cv2.destroyAllWindows()

    # =====================================================================
    # POST-RUN EMPIRICAL EVALUATION REPORT & PLOTTING
    # =====================================================================
    if len(metrics_log) > 10:
        df = pd.DataFrame(metrics_log)
        csv_filename = "live_video_metrics.csv"
        df.to_csv(csv_filename, index=False)

        total_nv_mb = df['NetVision_KB'].sum() / 1024.0
        total_std_mb = df['Baseline_KB'].sum() / 1024.0
        avg_savings = df['Bandwidth_Saved_Pct'].mean()
        avg_latency = df['Latency_ms'].mean()
        total_static_hits = df['Static_Cache_Hits'].sum()
        total_deltas = df['Moving_Deltas'].sum()

        print("\n" + "="*65)
        print("         NETVISION EMPIRICAL EVALUATION REPORT CARD         ")
        print("="*65)
        print(f" Total Frames Streamed:       {len(df)}")
        print(f" Standard Streaming Traffic:   {total_std_mb:.2f} MB")
        print(f" NetVision Streaming Traffic:  {total_nv_mb:.2f} MB")
        print(f" Net Bandwidth Reduction:     {avg_savings:.2f}% SAVED")
        print(f" Average Latency:             {avg_latency:.2f} ms")
        print(f" Zero-Bandwidth Cache Hits:   {total_static_hits} instances")
        print(f" Active Semantic Deltas:      {total_deltas} instances")
        print("="*65)
        print(f"[✓] Saved raw frame data to '{csv_filename}'")

        # Generate 3-panel publication figure
        plt.figure(figsize=(12, 8))

        # Subplot 1: Frame-by-Frame Payload Comparison
        plt.subplot(3, 1, 1)
        plt.plot(df['Frame'], df['Baseline_KB'], label='Standard Stream (Uniform JPEG Q=75)', color='red', linestyle='--', alpha=0.7)
        plt.plot(df['Frame'], df['NetVision_KB'], label='NetVision (Proposed Deltas)', color='blue', linewidth=1.8)
        plt.title("Empirical Payload Size Comparison per Frame", fontsize=11, fontweight='bold')
        plt.ylabel("Payload (KB)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc='upper right')

        # Subplot 2: Cumulative Bandwidth Saved (%)
        plt.subplot(3, 1, 2)
        plt.plot(df['Frame'], df['Bandwidth_Saved_Pct'], color='green', linewidth=1.8)
        plt.axhline(avg_savings, color='darkgreen', linestyle=':', label=f"Mean Savings ({avg_savings:.1f}%)")
        plt.title("Instantaneous Bandwidth Savings (%)", fontsize=11, fontweight='bold')
        plt.ylabel("Savings (%)")
        plt.ylim(0, 100)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc='lower right')

        # Subplot 3: Moving Deltas vs. Static 0-Byte Cache Hits
        plt.subplot(3, 1, 3)
        plt.bar(df['Frame'] - 0.2, df['Moving_Deltas'], width=0.4, label='Moving Deltas (Sent)', color='orange')
        plt.bar(df['Frame'] + 0.2, df['Static_Cache_Hits'], width=0.4, label='Static Cache (0 Bytes)', color='cyan')
        plt.title("Temporal Target Activity: Active Patches vs. Static Hits", fontsize=11, fontweight='bold')
        plt.xlabel("Frame Sequence Index")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc='upper right')

        plt.tight_layout()
        plot_filename = "live_video_plots.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"[✓] Generated research evaluation dashboard: '{plot_filename}'")

        # Open in Mac Preview automatically
        os.system(f"open {plot_filename}")

if __name__ == "__main__":
    main()