# NetVision Video Socket Demo

A single-process video streaming demo that sends compressed object-detection
deltas between a local TCP sender and receiver. It compares the transmitted
payload with a full-frame JPEG baseline and saves per-frame metrics and a plot.

## Requirements

- Python 3.10 or newer
- macOS for camera capture (the script uses OpenCV's AVFoundation backend)
- A webcam, or no webcam (the demo falls back to a synthetic stream)

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run

From this directory, start the demo:

```bash
python3 video_demo_sockets.py
```

The first run downloads the `yolov8n.pt` model through Ultralytics if it is not
already available. Press `q` in the video window to stop. After at least 11
frames, the program writes `live_video_metrics.csv` and
`live_video_plots.png` to the current directory. On macOS, it also opens the
plot in Preview.

## Files

- `video_demo_sockets.py` — sender, receiver, rendering, and metrics.
- `map_net.py` — priority scoring and JPEG quality-tier selection used by the
  sender.
- `requirements.txt` — Python packages required to run the demo.

## Network and data notes

The sender and receiver run in the same process and bind to `127.0.0.1:9999`;
the demo is not configured for streaming to another machine. It serializes
local payloads with Python `pickle`. Do not change the binding to accept
untrusted network peers, because unpickling data from an untrusted source can
execute arbitrary code. Video frames are processed locally; the demo does not
upload them.

This repository contains only the runnable demo and its direct support files.
