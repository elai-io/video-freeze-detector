import os
import cv2
import numpy as np
import argparse
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Combine first frames from video sets into single images')
    parser.add_argument('input_dir', help='Input directory containing video folders')
    parser.add_argument('--output', default='results', help='Output directory (default: results)')
    parser.add_argument('--crop-fraction', type=float, default=0.5,
                       help='Fraction of width for center crop (default: 0.5)')
    parser.add_argument('--line-position', type=float, default=0.15,
                       help='Position of bottom line as fraction from bottom (default: 0.15)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()


def find_video_folders(input_dir: str) -> List[str]:
    """Find folders containing exactly 3 video files (MOV, AVI, MP4)."""
    video_folders = []
    video_extensions = {'.mov', '.avi', '.mp4', '.mkv', '.wmv', '.flv', '.webm'}
    
    for root, dirs, files in os.walk(input_dir):
        video_files = [f for f in files if Path(f).suffix.lower() in video_extensions]
        if len(video_files) == 3:
            video_folders.append(root)
    
    return video_folders


def extract_first_frame(video_path: str) -> Optional[np.ndarray]:
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def process_frame(frame: np.ndarray, crop_fraction: float, line_position: float) -> Image.Image:
    """Process frame: crop center, add crosshair and bottom line."""
    h, w = frame.shape[:2]
    
    # Crop center 50% by width
    crop_w = int(w * crop_fraction)
    start_w = (w - crop_w) // 2
    cropped_frame = frame[:, start_w:start_w + crop_w]
    
    # Convert to PIL Image for drawing
    pil_image = Image.fromarray(cropped_frame)
    draw = ImageDraw.Draw(pil_image)
    
    # Get dimensions of cropped image
    crop_h, crop_w = cropped_frame.shape[:2]
    
    # Draw crosshair in center
    center_x, center_y = crop_w // 2, crop_h // 2
    crosshair_size = min(crop_w, crop_h) // 20  # 5% of smaller dimension
    
    # Horizontal line of crosshair
    draw.line([
        (center_x - crosshair_size, center_y),
        (center_x + crosshair_size, center_y)
    ], fill='red', width=3)
    
    # Vertical line of crosshair
    draw.line([
        (center_x, center_y - crosshair_size),
        (center_x, center_y + crosshair_size)
    ], fill='red', width=3)
    
    # Draw bottom line at 15% from bottom
    line_y = int(crop_h * (1 - line_position))
    draw.line([
        (0, line_y),
        (crop_w, line_y)
    ], fill='blue', width=3)
    
    return pil_image


def combine_images(images: List[Image.Image]) -> Image.Image:
    """Combine 3 images horizontally into one."""
    if len(images) != 3:
        raise ValueError("Expected exactly 3 images")
    
    # Get dimensions (assuming all images have same height after processing)
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    # Create combined image
    total_width = sum(widths)
    max_height = max(heights)
    
    combined = Image.new('RGB', (total_width, max_height), 'white')
    
    x_offset = 0
    for img in images:
        # Center image vertically if heights differ
        y_offset = (max_height - img.height) // 2
        combined.paste(img, (x_offset, y_offset))
        x_offset += img.width
    
    return combined


def process_video_folder(folder_path: str, crop_fraction: float, line_position: float, verbose: bool = False) -> Optional[Image.Image]:
    """Process a folder with 3 video files and return combined image."""
    if verbose:
        print(f"Processing folder: {folder_path}")
    
    # Find video files
    video_extensions = {'.mov', '.avi', '.mp4', '.mkv', '.wmv', '.flv', '.webm'}
    video_files = []
    for file in os.listdir(folder_path):
        if Path(file).suffix.lower() in video_extensions:
            video_files.append(os.path.join(folder_path, file))
    
    if len(video_files) != 3:
        if verbose:
            print(f"Warning: Found {len(video_files)} video files in {folder_path}, expected 3")
        return None
    
    # Sort files for consistent order (left, center, right based on filename)
    video_files.sort()
    
    processed_images = []
    for video_path in video_files:
        # Extract first frame
        frame = extract_first_frame(video_path)
        if frame is None:
            if verbose:
                print(f"Error: Could not extract frame from {video_path}")
            return None
        
        # Process frame
        processed_image = process_frame(frame, crop_fraction, line_position)
        processed_images.append(processed_image)
    
    # Combine images
    return combine_images(processed_images)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Convert input directory to absolute path
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Create output directory structure
    input_name = os.path.basename(input_dir.rstrip(os.sep))
    output_base = os.path.abspath(args.output)
    output_dir = os.path.join(output_base, input_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.verbose:
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
    
    # Find all video folders
    video_folders = find_video_folders(input_dir)
    
    if not video_folders:
        print("No folders with exactly 3 video files found")
        return
    
    if args.verbose:
        print(f"Found {len(video_folders)} video folders to process")
    
    # Process each folder
    for folder_path in tqdm(video_folders, desc="Processing video folders"):
        folder_name = os.path.basename(folder_path)
        
        # Process the folder
        combined_image = process_video_folder(
            folder_path, 
            args.crop_fraction, 
            args.line_position, 
            args.verbose
        )
        
        if combined_image is not None:
            # Save the combined image
            output_filename = f"{folder_name}.png"
            output_path = os.path.join(output_dir, output_filename)
            combined_image.save(output_path)
            
            if args.verbose:
                print(f"Saved: {output_path}")
        else:
            print(f"Error processing folder: {folder_path}")
    
    print(f"Processing complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 