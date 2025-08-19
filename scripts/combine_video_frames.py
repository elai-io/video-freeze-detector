import os
import cv2
import numpy as np
import argparse
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
import math


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Combine first frames from video sets into single images')
    parser.add_argument('input_dir', help='Input directory containing video folders')
    parser.add_argument('--output', default='results', help='Output directory (default: results)')
    parser.add_argument('--crop-fraction', type=float, default=0.4,
                       help='Fraction of width for center crop (default: 0.4)')
    parser.add_argument('--line-position', type=float, default=0.15,
                       help='Position of top and bottom lines as fraction from edges (default: 0.15)')
    parser.add_argument('--no-collage', action='store_true',
                       help='Skip creating collages (by default both full and 8K collages are created)')
    parser.add_argument('--create-collage', action='store_true',
                       help='Create a large collage from all processed images (default: enabled)')
    parser.add_argument('--collage-8k', action='store_true',
                       help='Also create an 8K version of the collage (default: enabled)')
    parser.add_argument('--frame-index', type=int, default=0,
                       help='Zero-based index of frame to extract from each video (default: 0 = first frame)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()


def _parse_unique_key_from_stem(stem: str) -> str:
    """Extract unique key from filename stem by dropping first two underscore tokens.

    Example: 'M006_frontal_angry_high_001' -> 'angry_high_001'
    Falls back to full stem if not enough tokens.
    """
    parts = stem.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[2:])
    return stem


def _list_video_files(directory: str) -> List[str]:
    """List video files in the given directory with supported extensions."""
    video_extensions = {'.mov', '.avi', '.mp4', '.mkv', '.wmv', '.flv', '.webm'}
    files: List[str] = []
    if not os.path.isdir(directory):
        return files
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and Path(name).suffix.lower() in video_extensions:
            files.append(path)
    return files


def find_strict_triplets(base_dir: str) -> List[Tuple[List[str], str]]:
    """Find matching triplets across front/left/right subfolders using unique suffix key.

    Returns a list of (paths, label) where paths is [left, front, right] order,
    and label is taken from the front filename stem (full, without extension).
    """
    cams = {
        'left': os.path.join(base_dir, 'left'),
        'front': os.path.join(base_dir, 'front'),
        'right': os.path.join(base_dir, 'right'),
    }

    if not all(os.path.isdir(p) for p in cams.values()):
        return []

    # Build key -> file map per camera
    maps: dict = {}
    for cam, folder in cams.items():
        mapping = {}
        for path in _list_video_files(folder):
            stem = Path(path).stem
            key = _parse_unique_key_from_stem(stem)
            mapping[key] = path
        maps[cam] = mapping

    # Keys present in all three
    common_keys = set(maps['left'].keys()) & set(maps['front'].keys()) & set(maps['right'].keys())
    if not common_keys:
        return []

    triplets: List[Tuple[List[str], str]] = []
    for key in sorted(common_keys):
        left_path = maps['left'][key]
        front_path = maps['front'][key]
        right_path = maps['right'][key]
        # Label by the front filename (stem)
        label = Path(front_path).stem
        triplets.append(([left_path, front_path, right_path], label))
    return triplets


def extract_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """Extract a specific frame by index from a video file (0-based)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Try to seek to the requested frame
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    cap.release()

    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def process_frame(frame: np.ndarray, crop_fraction: float, line_position: float) -> Image.Image:
    """Process frame: crop center, add measurement scale and top/bottom lines."""
    h, w = frame.shape[:2]
    
    # Crop center by specified fraction of width
    crop_w = int(w * crop_fraction)
    start_w = (w - crop_w) // 2
    cropped_frame = frame[:, start_w:start_w + crop_w]
    
    # Convert to PIL Image for drawing
    pil_image = Image.fromarray(cropped_frame)
    draw = ImageDraw.Draw(pil_image)
    
    # Get dimensions of cropped image
    crop_h, crop_w = cropped_frame.shape[:2]
    
    # Draw measurement scale in center
    center_x, center_y = crop_w // 2, crop_h // 2
    
    # Horizontal scale line (600 pixels total: 300 in each direction)
    scale_length = 300  # pixels in each direction from center
    draw.line([
        (center_x - scale_length, center_y),
        (center_x + scale_length, center_y)
    ], fill='red', width=2)
    
    # Scale marks every 50 pixels, with 100-pixel marks being longer
    for offset in range(-scale_length, scale_length + 1, 50):
        if offset == 0:
            continue  # Skip center mark, will draw vertical line
        
        mark_x = center_x + offset
        # Longer marks every 100 pixels
        if offset % 100 == 0:
            mark_height = 15  # Long mark
        else:
            mark_height = 8   # Short mark
        
        # Draw scale mark
        draw.line([
            (mark_x, center_y - mark_height),
            (mark_x, center_y + mark_height)
        ], fill='red', width=2)
    
    # Central vertical line (100 pixels total: 50 up and 50 down)
    vertical_length = 50  # pixels in each direction from center
    draw.line([
        (center_x, center_y - vertical_length),
        (center_x, center_y + vertical_length)
    ], fill='red', width=2)
    
    # Draw bottom line at 15% from bottom
    line_y = int(crop_h * (1 - line_position))
    draw.line([
        (0, line_y),
        (crop_w, line_y)
    ], fill='blue', width=3)
    
    # Draw top line at 15% from top
    top_line_y = int(crop_h * line_position)
    draw.line([
        (0, top_line_y),
        (crop_w, top_line_y)
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


def calculate_optimal_grid(num_images: int, img_width: int, img_height: int) -> Tuple[int, int]:
    """Calculate grid layout with fixed 12 columns per row."""
    cols = 12  # Fixed number of columns
    rows = math.ceil(num_images / cols)
    
    return cols, rows


def create_collage(images_data: List[Tuple[Image.Image, str]], output_path: str, 
                  scale_factor: float = 1.0, verbose: bool = False) -> None:
    """Create a large collage from all images with labels."""
    if not images_data:
        print("No images to create collage")
        return
    
    # Get first image dimensions
    first_img = images_data[0][0]
    img_width, img_height = first_img.size
    
    # Scale dimensions if needed
    if scale_factor != 1.0:
        img_width = int(img_width * scale_factor)
        img_height = int(img_height * scale_factor)
    
    # Calculate optimal grid
    cols, rows = calculate_optimal_grid(len(images_data), img_width, img_height)
    
    if verbose:
        print(f"Creating collage with {cols}×{rows} grid")
        print(f"Image size: {img_width}×{img_height}")
        print(f"Total images: {len(images_data)}")
    
    # Calculate collage dimensions
    collage_width = cols * img_width
    collage_height = rows * img_height
    
    # Create black background
    collage = Image.new('RGB', (collage_width, collage_height), 'black')
    
    # Try to load a font for labels
    try:
        # Try to use a system font
        font_size = max(12, int(img_height * 0.03))  # 3% of image height
        if scale_factor != 1.0:
            font_size = int(font_size * scale_factor)
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.load_default()
        except (OSError, IOError):
            font = None
    
    # Place images in grid
    for i, (img, label) in enumerate(tqdm(images_data, desc="Creating collage", disable=not verbose)):
        row = i // cols
        col = i % cols
        
        # Calculate position
        x = col * img_width
        y = row * img_height
        
        # Resize image if needed
        if scale_factor != 1.0:
            img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        
        # Paste image
        collage.paste(img, (x, y))
        
        # Add label if font is available
        if font:
            draw = ImageDraw.Draw(collage)
            
            # Calculate text position (top-left corner with some padding)
            text_x = x + 10
            text_y = y + 10
            
            # Draw text with outline for better visibility
            outline_width = 2
            text_color = 'white'
            outline_color = 'black'
            
            # Draw outline
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, text_y + dy), label, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((text_x, text_y), label, font=font, fill=text_color)
    
    # Save collage
    if verbose:
        print(f"Saving collage: {collage_width}×{collage_height} pixels")
    
    collage.save(output_path, 'JPEG', quality=95, optimize=True)
    
    if verbose:
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Collage saved: {output_path} ({file_size:.1f} MB)")


def process_video_triplet(video_paths: List[str], crop_fraction: float, line_position: float, frame_index: int) -> Optional[Image.Image]:
    """Process explicit list of 3 video paths and return combined image."""
    if len(video_paths) != 3:
        return None

    processed_images: List[Image.Image] = []
    for video_path in video_paths:
        frame = extract_frame(video_path, frame_index)
        if frame is None:
            return None
        processed_image = process_frame(frame, crop_fraction, line_position)
        processed_images.append(processed_image)

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
    video_folders = find_strict_triplets(input_dir)
    
    if not video_folders:
        print("No folders with exactly 3 video files found")
        return
    
    if args.verbose:
        print(f"Found {len(video_folders)} video folders to process")
    
    # Store all processed images for collage
    all_images = []
    
    # Process each folder
    for folder_path, label in tqdm(video_folders, desc="Processing video folders"):
        output_filename = f"{label}.png"
        output_path = os.path.join(output_dir, output_filename)

        # If output already exists, skip processing and load for collage
        if os.path.exists(output_path):
            if args.verbose:
                print(f"Skipping existing: {output_path}")
            if not args.no_collage:
                try:
                    existing_img = Image.open(output_path).convert('RGB')
                    all_images.append((existing_img, label))
                except Exception as exc:
                    print(f"Warning: could not open existing image '{output_path}': {exc}")
            continue

        # Process the folder normally
        combined_image = process_video_triplet(
            folder_path,
            args.crop_fraction,
            args.line_position,
            args.frame_index,
        )
        
        if combined_image is not None:
            # Save the combined image
            combined_image.save(output_path)
            
            # Store for collage (by default both are enabled unless --no-collage is specified)
            if not args.no_collage:
                all_images.append((combined_image, label))
            
            if args.verbose:
                print(f"Saved: {output_path}")
        else:
            print(f"Error processing folder: {folder_path}")
    
    print(f"Processing complete. Results saved to: {output_dir}")
    
    # Create collages (by default both are created unless --no-collage is specified)
    if not args.no_collage and all_images:
        print(f"\nCreating collages from {len(all_images)} images...")
        
        # Determine which collages to create
        create_full = args.create_collage or not args.no_collage  # Default to True
        create_8k = args.collage_8k or not args.no_collage       # Default to True
        
        if create_full:
            collage_path = os.path.join(output_dir, "collage_full.jpg")
            print("Creating full resolution collage...")
            create_collage(all_images, collage_path, scale_factor=1.0, verbose=args.verbose)
        
        if create_8k:
            # Calculate scale factor for 8K version (fixed 12 columns)
            if all_images:
                first_img = all_images[0][0]
                original_img_width = first_img.width
                
                # Fixed grid: 12 columns
                cols = 12
                rows = math.ceil(len(all_images) / cols)
                
                # Calculate scale factor to fit 8K width (7680 pixels)
                target_width = 7680  # 8K width
                scale_factor = target_width / (cols * original_img_width)
                
                collage_8k_path = os.path.join(output_dir, "collage_8k.jpg")
                print(f"Creating 8K collage with {cols}×{rows} grid (scale factor: {scale_factor:.3f})...")
                create_collage(all_images, collage_8k_path, scale_factor=scale_factor, verbose=args.verbose)
    elif not args.no_collage:
        print("No images available for collage creation. Ensure per-triplet PNGs exist.")


if __name__ == "__main__":
    main() 