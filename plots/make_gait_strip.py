"""
Build a side-by-side gait-cycle comparison figure from the three play_best.mp4
videos (one full ~0.8s gait cycle per model, 5 frames @ 50fps spaced 0.2s apart).

Usage:
    python make_gait_strip.py
"""
import os
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

VIDEOS = {
    'v4_r48s':        '../videos/videos_v4_r48s/play_best.mp4',
    'v4_r48s_acc5x':  '../videos/videos_v4_r48s_acc5x/play_best.mp4',
    'v4_r48s_acc10x': '../videos/videos_v4_r48s_acc10x/play_best.mp4',
}
FRAME_IDS = [210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260]   # t = 0.0..1.8s @ 0.2s apart (50Hz)
CROP      = (560, 760, 100, 580)        # x0, x1, y0, y1 — robot-centered window
LABEL_W   = 210                          # left label column width (px)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(ROOT, 'report', 'figs', 'gait_comparison.png')


def brighten(img, gamma=0.6, gain=1.25):
    x = img.astype(np.float32) / 255.0
    x = np.clip(x * gain, 0.0, 1.0) ** gamma
    return (x * 255).astype(np.uint8)


def grab_frames(path, ids):
    r = imageio.get_reader(path)
    out = {}
    for i, frame in enumerate(r):
        if i in ids:
            out[i] = frame
        if i >= max(ids):
            break
    r.close()
    return [out[i] for i in ids]


def label_row(name, row_img, h):
    """Prepend a vertical label column with the model name."""
    label = Image.new('RGB', (LABEL_W, h), (255, 255, 255))
    draw = ImageFont
    d = ImageDraw.Draw(label)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 38)
    except OSError:
        font = ImageFont.load_default()
    text = name.replace('_', '\n')
    bbox = d.multiline_textbbox((0, 0), text, font=font, align='center')
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.multiline_text(((LABEL_W - tw) / 2, (h - th) / 2), text,
                      font=font, fill=(0, 0, 0), align='center')
    return np.concatenate([np.array(label), row_img], axis=1)


def time_ticks(strip_img, n_frames, frame_w, x_offset, crop_h):
    """Overlay t=0.0s ... labels directly onto the bottom edge of the last
    row's frames (white text with a black outline so it stays legible
    against both the dark sky and the light floor)."""
    img = Image.fromarray(strip_img)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
    except OSError:
        font = ImageFont.load_default()
    last_row_bottom = img.height
    for k in range(n_frames):
        t = k * 0.1
        text = f"{t:.1f}s"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx = x_offset + k * frame_w + frame_w // 2
        y  = last_row_bottom - th - 14
        draw.text((cx - tw / 2, y), text, font=font, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(img)


def main():
    rows = []
    crop_h = CROP[3] - CROP[2]
    crop_w = CROP[1] - CROP[0]

    for name, rel_path in VIDEOS.items():
        path = os.path.join(HERE, rel_path)
        frames = grab_frames(path, FRAME_IDS)
        crops = [brighten(f[CROP[2]:CROP[3], CROP[0]:CROP[1]]) for f in frames]
        row = np.concatenate(crops, axis=1)
        rows.append(label_row(name, row, crop_h))

    # Stack rows, add a thin separator between them
    sep = np.full((4, rows[0].shape[1], 3), 200, dtype=np.uint8)
    body = rows[0]
    for r in rows[1:]:
        body = np.concatenate([body, sep, r], axis=0)

    # Overlay time ticks directly onto the bottom row's frames (no extra margin)
    composite = time_ticks(body, len(FRAME_IDS), crop_w, LABEL_W, crop_h)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.imwrite(OUT, composite)
    print(f"Saved → {OUT}  shape={composite.shape}")


if __name__ == '__main__':
    main()
