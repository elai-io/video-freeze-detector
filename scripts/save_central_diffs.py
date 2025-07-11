import os
import cv2
import numpy as np
import argparse
import subprocess
from typing import List
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

CONTRAST_FACTOR = 2.0  # Усиление контраста
CROP_FRACTION = 1/3    # Центральная треть по ширине
FONT_SIZE = 32
FPS = 5  # Частота кадров для выходного видео

# Попробуем найти шрифт для PIL
try:
    FONT = ImageFont.truetype("arial.ttf", FONT_SIZE)
except Exception:
    FONT = ImageFont.load_default()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process video files and create difference images')
    parser.add_argument('input_dir', help='Input directory containing video files')
    parser.add_argument('output_dir', help='Output directory for difference images and video')
    parser.add_argument('--fps', type=int, default=FPS, 
                       help=f'FPS for output video (default: {FPS})')
    parser.add_argument('--contrast', type=float, default=CONTRAST_FACTOR,
                       help=f'Contrast factor for difference images (default: {CONTRAST_FACTOR})')
    parser.add_argument('--crop-fraction', type=float, default=CROP_FRACTION,
                       help=f'Central crop fraction (default: {CROP_FRACTION})')
    return parser.parse_args()


def get_video_files(input_dir: str) -> List[str]:
    """Get list of video files from input directory."""
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]


def central_crop(img: np.ndarray, fraction: float = 1/3) -> np.ndarray:
    """Crop central portion of image."""
    h, w = img.shape[:2]
    start = int(w * (0.5 - fraction/2))
    end = int(w * (0.5 + fraction/2))
    return img[:, start:end]


def save_diff_image(diff: np.ndarray, frame_idx: int, mean_diffs: List[float], out_path: str):
    """Save difference image with annotations."""
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
    draw.text((10, 10), font=FONT, fill=(255, 255, 0), text=text)
    img_pil.save(out_path)


def create_video_from_images(image_dir: str, output_video_path: str, fps: int) -> bool:
    """Create video from sequence of images using ffmpeg."""
    try:
        # Формируем паттерн для поиска изображений
        image_pattern = os.path.join(image_dir, "diff_%04d.png")
        
        # Команда ffmpeg для создания видео
        cmd = [
            'ffmpeg',
            '-y',  # Перезаписать выходной файл если существует
            '-framerate', str(fps),
            '-i', image_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',  # Качество сжатия
            output_video_path
        ]
        
        print(f"Creating video with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Video created successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg and add it to PATH.")
        return False


def main():
    # Парсим аргументы командной строки
    args = parse_arguments()
    
    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Получаем список видео файлов
    video_files = get_video_files(args.input_dir)
    if not video_files:
        print(f"No video files found in {args.input_dir}")
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
            diffs_cropped = [central_crop(d, args.crop_fraction) for d in diffs]
            # Усреднение для вывода
            mean_diffs = [float(np.mean(d)) for d in diffs_cropped]
            # Объединяем по ширине
            diff_concat = np.concatenate(diffs_cropped, axis=1)
            # diff[0] соответствует кадру 1 (разность между кадрами 0 и 1)
            diff_frame_idx = frame_idx
            out_path = os.path.join(args.output_dir, f'diff_{diff_frame_idx:04d}.png')
            save_diff_image(diff_concat, diff_frame_idx, mean_diffs, out_path)
            pbar.update(1)
        
        prev_frames = frames_gray
        frame_idx += 1
    
    pbar.close()
    for cap in caps:
        cap.release()
    
    print(f"Done! Saved diffs to {args.output_dir}")
    
    # Создаем видео из изображений
    output_video_path = os.path.join(args.output_dir, "diffs_video.mp4")
    print(f"Creating video from images...")
    if create_video_from_images(args.output_dir, output_video_path, args.fps):
        print(f"Video saved to: {output_video_path}")
    else:
        print("Failed to create video.")


if __name__ == '__main__':
    main() 