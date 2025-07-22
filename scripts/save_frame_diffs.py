import os
import cv2
import numpy as np
import argparse
from typing import Optional, List
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt

CONTRAST_FACTOR = 2.0  # Усиление контраста
FONT_SIZE = 32
CROP_FRACTION = 0.5  # Доля ширины для центрального кропа (квадрат)

# Попробуем найти шрифт для PIL
try:
    FONT = ImageFont.truetype("arial.ttf", FONT_SIZE)
except Exception:
    FONT = ImageFont.load_default()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract frame differences from a single video file')
    parser.add_argument('video_file', help='Path to video file (MP4, AVI, MOV, MKV)')
    parser.add_argument('output_dir', help='Output directory for difference images')
    parser.add_argument('--contrast', type=float, default=CONTRAST_FACTOR,
                       help=f'Contrast factor for difference images (default: {CONTRAST_FACTOR})')
    parser.add_argument('--fps', type=float, default=5.0,
                       help='FPS for output video (default: 5.0)')
    parser.add_argument('--crop-fraction', type=float, default=CROP_FRACTION,
                       help=f'Fraction of width for center crop (creates square, default: {CROP_FRACTION})')
    parser.add_argument('--no-plot', action='store_true', help='Skip saving the plot of mean differences')
    parser.add_argument('--no-video', action='store_true', help='Skip creating output video')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()


def is_video_file(file_path: str) -> bool:
    """Check if file is a supported video format."""
    return file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))


def get_center_crop(frame: np.ndarray, crop_fraction: float) -> np.ndarray:
    """Extract center crop from frame - crop_fraction applies to width only."""
    h, w = frame.shape[:2]
    crop_w = int(w * crop_fraction)  # Кроп по ширине
    crop_h = crop_w  # Кроп по высоте равен кропу по ширине (квадрат)
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return frame[start_h:start_h + crop_h, start_w:start_w + crop_w]


def get_edges(frame_gray: np.ndarray) -> np.ndarray:
    """Extract edges from grayscale frame using Canny edge detector."""
    # Применяем Gaussian blur для уменьшения шума
    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    
    # Применяем Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges


def save_side_by_side_image(original_crop: np.ndarray, diff_crop: np.ndarray, frame_idx: int, 
                           mean_diff: float, out_path: str, contrast_factor: float):
    """Save side-by-side image with original crop on left and difference on right."""
    # Усиливаем контраст для разности
    diff_vis = np.clip(diff_crop * contrast_factor, 0, 255).astype(np.uint8)
    
    # Переводим в RGB если нужно
    if diff_vis.ndim == 2:
        diff_vis = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2RGB)
    elif diff_vis.shape[2] == 1:
        diff_vis = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2RGB)
    
    # Переводим оригинал в RGB если нужно
    if original_crop.ndim == 2:
        original_rgb = cv2.cvtColor(original_crop, cv2.COLOR_GRAY2RGB)
    elif original_crop.shape[2] == 3:
        original_rgb = cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original_crop
    
    # Объединяем изображения горизонтально
    combined = np.hstack([original_rgb, diff_vis])
    
    # PIL для текста
    img_pil = Image.fromarray(combined)
    draw = ImageDraw.Draw(img_pil)
    
    # Добавляем подписи
    draw.text((10, 10), "Original", font=FONT, fill=(255, 255, 0))
    draw.text((original_rgb.shape[1] + 10, 10), "Difference", font=FONT, fill=(255, 255, 0))
    
    # Добавляем информацию о кадре
    info_text = f"Frame: {frame_idx} | Mean diff: {mean_diff:.2f}"
    text_bbox = draw.textbbox((0, 0), info_text, font=FONT)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (combined.shape[1] - text_width) // 2
    draw.text((text_x, combined.shape[0] - 50), info_text, font=FONT, fill=(255, 255, 0))
    
    img_pil.save(out_path)


def create_video_from_images(image_dir: str, output_video_path: str, fps: float):
    """Create video from saved images."""
    # Получаем список всех изображений
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and f.startswith('diff_')]
    image_files.sort()
    
    if not image_files:
        print("Warning: No images found to create video")
        return
    
    # Читаем первое изображение для получения размеров
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print("Error: Cannot read first image")
        return
    
    height, width = first_image.shape[:2]
    
    # Создаем видеописатель
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Cannot create video writer")
        return
    
    print(f"🎬 Creating video from {len(image_files)} images...")
    
    # Добавляем кадры в видео
    for image_file in tqdm(image_files, desc="Creating video", unit="frame"):
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    print(f"✅ Video saved to: {output_video_path}")


def save_plots(mean_diffs: List[float], edge_diffs: List[float], fps: float, output_dir: str, video_filename: str):
    """Save plots of mean differences and edge differences over time."""
    if not mean_diffs:
        print("Warning: No data to plot")
        return
    
    # Создаем временную ось (в секундах)
    time_axis = [i / fps for i in range(len(mean_diffs))]
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # График 1: Обычные разности кадров
    ax1.plot(time_axis, mean_diffs, linewidth=1, alpha=0.7, color='blue')
    ax1.set_title(f'Frame Differences Over Time\nVideo: {video_filename}', fontsize=14, pad=20)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Mean Frame Difference', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Статистика для обычных разностей
    mean_val = np.mean(mean_diffs)
    std_val = np.std(mean_diffs)
    max_val = np.max(mean_diffs)
    min_val = np.min(mean_diffs)
    
    # Горизонтальные линии для статистики
    ax1.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
    ax1.axhline(y=mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'Mean + Std: {mean_val + std_val:.2f}')
    ax1.axhline(y=mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'Mean - Std: {mean_val - std_val:.2f}')
    
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, max(time_axis))
    ax1.set_ylim(0, max_val * 1.1)
    
    # Статистика в текстовом блоке для первого графика
    stats_text = f'Statistics:\nMean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nFrames: {len(mean_diffs)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # График 2: Разности границ (edges)
    if edge_diffs:
        ax2.plot(time_axis, edge_diffs, linewidth=1, alpha=0.7, color='green')
        ax2.set_title('Edge Differences Over Time', fontsize=14, pad=20)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Mean Edge Difference', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Статистика для разностей границ
        edge_mean_val = np.mean(edge_diffs)
        edge_std_val = np.std(edge_diffs)
        edge_max_val = np.max(edge_diffs)
        edge_min_val = np.min(edge_diffs)
        
        # Горизонтальные линии для статистики
        ax2.axhline(y=edge_mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {edge_mean_val:.2f}')
        ax2.axhline(y=edge_mean_val + edge_std_val, color='orange', linestyle=':', alpha=0.7, label=f'Mean + Std: {edge_mean_val + edge_std_val:.2f}')
        ax2.axhline(y=edge_mean_val - edge_std_val, color='orange', linestyle=':', alpha=0.7, label=f'Mean - Std: {edge_mean_val - edge_std_val:.2f}')
        
        ax2.legend(loc='upper right')
        ax2.set_xlim(0, max(time_axis))
        ax2.set_ylim(0, edge_max_val * 1.1)
        
        # Статистика в текстовом блоке для второго графика
        edge_stats_text = f'Edge Statistics:\nMean: {edge_mean_val:.2f}\nStd: {edge_std_val:.2f}\nMin: {edge_min_val:.2f}\nMax: {edge_max_val:.2f}'
        ax2.text(0.02, 0.98, edge_stats_text, transform=ax2.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No edge data available', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=14, alpha=0.5)
        ax2.set_title('Edge Differences Over Time', fontsize=14, pad=20)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Mean Edge Difference', fontsize=12)
    
    # Сохраняем график
    plot_path = os.path.join(output_dir, 'frame_differences_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Combined plot saved to: {plot_path}")
    
    # Сохраняем данные в CSV
    csv_path = os.path.join(output_dir, 'frame_differences_data.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('Frame,Time_Seconds,Mean_Difference,Edge_Difference\n')
        for i, (time, diff, edge_diff) in enumerate(zip(time_axis, mean_diffs, edge_diffs or [0]*len(mean_diffs))):
            f.write(f'{i},{time:.3f},{diff:.3f},{edge_diff:.3f}\n')
    
    print(f"📊 Data saved to: {csv_path}")


def main():
    # Парсим аргументы командной строки
    args = parse_arguments()
    
    # Проверяем видео файл
    if not os.path.exists(args.video_file):
        print(f"Error: Video file {args.video_file} not found")
        return 1
    
    if not is_video_file(args.video_file):
        print(f"Error: {args.video_file} is not a supported video format (MP4, AVI, MOV, MKV)")
        return 1
    
    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Открываем видео файл
    cap = cv2.VideoCapture(args.video_file)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args.video_file}")
        return 1
    
    # Получаем информацию о видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("📹 VIDEO FRAME DIFFERENCE EXTRACTOR")
    print("=" * 50)
    print(f"Video file: {os.path.basename(args.video_file)}")
    print(f"Resolution: {width}x{height}")
    print(f"Original FPS: {original_fps:.2f}")
    print(f"Output FPS: {args.fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Output directory: {args.output_dir}")
    print(f"Contrast factor: {args.contrast}")
    print(f"Crop fraction: {args.crop_fraction}")
    print("-" * 50)
    
    if total_frames <= 1:
        print("Error: Video must have at least 2 frames to compute differences")
        cap.release()
        return 1
    
    # Сбрасываем позицию
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_idx = 0
    prev_frame: Optional[np.ndarray] = None
    diff_count = 0
    mean_diffs: List[float] = []  # Список для хранения средних значений разностей
    edge_diffs: List[float] = []  # Список для хранения разностей границ
    
    # Создаем прогресс-бар
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
    
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if prev_frame is not None:
            # Получаем центральные кропы
            current_crop = get_center_crop(frame, args.crop_fraction)
            prev_crop = get_center_crop(prev_frame, args.crop_fraction)
            
            # Конвертируем в grayscale для вычисления разности
            current_crop_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)
            prev_crop_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
            
            # Вычисляем разность кадров
            diff = cv2.absdiff(current_crop_gray, prev_crop_gray)
            
            # Вычисляем среднее значение разности
            mean_diff = float(np.mean(diff))
            mean_diffs.append(mean_diff)  # Добавляем в список для графика
            
            # Вычисляем разность границ
            current_edges = get_edges(current_crop_gray)
            prev_edges = get_edges(prev_crop_gray)
            edge_diff = cv2.absdiff(current_edges, prev_edges)
            mean_edge_diff = float(np.mean(edge_diff))
            edge_diffs.append(mean_edge_diff)
            
            # Сохраняем изображение с двумя кропами
            out_path = os.path.join(args.output_dir, f'diff_{diff_count:04d}.png')
            save_side_by_side_image(current_crop, diff, frame_idx, mean_diff, out_path, args.contrast)
            
            if args.verbose:
                print(f"Frame {frame_idx}: mean diff = {mean_diff:.2f}, edge diff = {mean_edge_diff:.2f}")
            
            diff_count += 1
        
        prev_frame = frame.copy()
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"\n✅ Done! Processed {frame_idx} frames")
    print(f"📁 Saved {diff_count} side-by-side images to {args.output_dir}")
    print(f"📊 Average frames per second: {original_fps:.2f}")
    print(f"📏 Image resolution: {width}x{height}")
    
    # Создаем видео из изображений
    if not args.no_video and diff_count > 0:
        video_path = os.path.join(args.output_dir, 'frame_differences_video.mp4')
        create_video_from_images(args.output_dir, video_path, args.fps)
    
    # Сохраняем графики средних значений разностей
    if not args.no_plot and mean_diffs:
        print("\n📈 Creating plots of frame differences...")
        video_filename = os.path.basename(args.video_file)
        save_plots(mean_diffs, edge_diffs, original_fps, args.output_dir, video_filename)
        
        # Выводим статистику для обычных разностей
        mean_val = np.mean(mean_diffs)
        std_val = np.std(mean_diffs)
        max_val = np.max(mean_diffs)
        min_val = np.min(mean_diffs)
        
        print("📊 Frame Difference Statistics:")
        print(f"   Mean: {mean_val:.3f}")
        print(f"   Std:  {std_val:.3f}")
        print(f"   Min:  {min_val:.3f}")
        print(f"   Max:  {max_val:.3f}")
        
        # Выводим статистику для разностей границ
        if edge_diffs:
            edge_mean_val = np.mean(edge_diffs)
            edge_std_val = np.std(edge_diffs)
            edge_max_val = np.max(edge_diffs)
            edge_min_val = np.min(edge_diffs)
            
            print("📊 Edge Difference Statistics:")
            print(f"   Mean: {edge_mean_val:.3f}")
            print(f"   Std:  {edge_std_val:.3f}")
            print(f"   Min:  {edge_min_val:.3f}")
            print(f"   Max:  {edge_max_val:.3f}")
    
    return 0


if __name__ == '__main__':
    exit(main()) 