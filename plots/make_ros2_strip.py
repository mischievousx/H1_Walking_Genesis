"""
Build a 2x3 key-frame strip from the v4_r48s ROS2 closed-loop route video
(videos/ros2_path/r48s.mp4), illustrating the full 6-segment command sequence:
forward -> left turn -> forward -> right turn -> forward -> stop.

Each frame already carries the play_ros2_path.py on-screen overlay
("[ROS2 closed-loop] Phase: ...", vx / step, heading actual/target/err),
which is exactly the /imu -> DemoSequenceNode -> /cmd_vel feedback being
illustrated — so frames are used as-is (no extra annotation needed), one
per route segment, sampled from the middle of that segment.

Usage:
    python make_ros2_strip.py
"""
import os

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

VIDEO = '../videos/ros2_path/r48s.mp4'

# (frame index, segment label) — mid-segment sample for each of the 6 route
# phases; segment boundaries are at cumulative steps [200,500,700,1000,1150,1250]
# (see FIXED_ROUTE in play_ros2_path.py).
FRAME_IDS = [
    (100,  '1. Forward'),
    (350,  '2. Left turn'),
    (600,  '3. Forward'),
    (850,  '4. Right turn'),
    (1075, '5. Forward'),
    (1200, '6. Stop'),
]
N_COLS  = 3
HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
OUT     = os.path.join(ROOT, 'report', 'figs', 'ros2_control_strip.png')


def grab_frames(path, ids):
    r = imageio.get_reader(path)
    out = {}
    targets = set(ids)
    for i, frame in enumerate(r):
        if i in targets:
            out[i] = frame
        if i >= max(ids):
            break
    r.close()
    return out


def label_frame(frame, text):
    """Draw a small bottom-left segment-order label badge onto the frame
    (the in-frame overlay already shows phase/heading — this just makes the
    segment ordering scannable at a glance in the strip)."""
    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = 14, img.height - th - 26
    draw.rectangle([x - 10, y - 6, x + tw + 10, y + th + 10], fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 0))
    return np.array(img)


def main():
    ids = [i for i, _ in FRAME_IDS]
    frames = grab_frames(os.path.join(HERE, VIDEO), ids)

    tiles = [label_frame(frames[i], label) for i, label in FRAME_IDS]
    h, w = tiles[0].shape[:2]
    n_rows = (len(tiles) + N_COLS - 1) // N_COLS

    sep = 4
    canvas = np.full((n_rows * h + (n_rows - 1) * sep,
                      N_COLS * w + (N_COLS - 1) * sep, 3), 255, dtype=np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, N_COLS)
        y0, x0 = r * (h + sep), c * (w + sep)
        canvas[y0:y0 + h, x0:x0 + w] = tile

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.imwrite(OUT, canvas)
    print(f"Saved -> {OUT}  shape={canvas.shape}")


if __name__ == '__main__':
    main()
