import os
import cv2
import numpy as np
from typing import List
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

INPUT_DIR = 'videos'
OUTPUT_DIR = 'output_diffs'
CONTRAST_FACTOR = 2.0  # Усиление контраста
CROP_FRACTION = 1/3    # Центральная треть по ширине
FONT_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Попробуем найти шрифт для PIL
try:
    FONT = ImageFont.truetype("arial.ttf", FONT_SIZE)
except Exception:
    FONT = ImageFont.load_default()

def get_video_files(input_dir: str) -> List[str]:
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

def central_crop(img: np.ndarray, fraction: float = 1/3) -> np.ndarray:
    h, w = img.shape[:2]
    start = int(w * (0.5 - fraction/2))
    end = int(w * (0.5 + fraction/2))
    return img[:, start:end]

def save_diff_image(diff: np.ndarray, frame_idx: int, mean_diffs: List[float], out_path: str):
    # frame_idx здесь - это номер кадра, для которого вычисляется разность
    # Усиливаем контраст
    diff_vis = np.clip(diff * CONTRAST_FACTOR, 0, 255).astype(np.uint8)
    # Переводим в RGB если нужно
    if diff_vis.ndim == 2:
        diff_vis = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2RGB)
    elif diff_vis.shape[2] == 1:
        diff_vis = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2RGB)
    # PIL для текста
    img_pil = Image.fromarray(diff_vis)
    draw = ImageDraw.Draw(img_pil)
    text = f"Frame: {frame_idx}\nMean diff: {['{:.2f}'.format(m) for m in mean_diffs]}"
    draw.text((10, 10), text, font=FONT, fill=(255, 255, 0))
    img_pil.save(out_path)

def main():
    video_files = get_video_files(INPUT_DIR)
    if not video_files:
        print(f"No video files found in {INPUT_DIR}")
        return
    print(f"Found {len(video_files)} video files.")
    
    # Определяем общее количество кадров
    caps = [cv2.VideoCapture(f) for f in video_files]
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
    print(f"Total frames to process: {total_frames}")
    
    # Сбрасываем позицию всех капчур
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_idx = 0  # Начинаем с 0
    prev_frames = []
    
    # Создаем прогресс-бар
    pbar = tqdm(total=total_frames-1, desc="Processing frames", unit="frame")
    
    while frame_idx < total_frames:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frames = None
                break
            frames.append(frame)
        if frames is None or len(frames) != len(caps):
            break
        # BGR -> Gray для диффа
        frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        if prev_frames:
            diffs = [cv2.absdiff(frames_gray[i], prev_frames[i]) for i in range(len(frames_gray))]
            # Кроп центральной трети
            diffs_cropped = [central_crop(d, CROP_FRACTION) for d in diffs]
            # Усреднение для вывода
            mean_diffs = [float(np.mean(d)) for d in diffs_cropped]
            # Объединяем по ширине
            diff_concat = np.concatenate(diffs_cropped, axis=1)
            # diff[0] соответствует кадру 1 (разность между кадрами 0 и 1)
            diff_frame_idx = frame_idx
            out_path = os.path.join(OUTPUT_DIR, f'diff_{diff_frame_idx:04d}.png')
            save_diff_image(diff_concat, diff_frame_idx, mean_diffs, out_path)
            pbar.update(1)
        prev_frames = frames_gray
        frame_idx += 1
    
    pbar.close()
    for cap in caps:
        cap.release()
    print(f"Done! Saved diffs to {OUTPUT_DIR}")

if __name__ == '__main__':
    main() 