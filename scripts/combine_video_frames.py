import os
import json
import subprocess
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
                       help='Skip creating collage image')
    parser.add_argument('--collage-cols', type=int, default=6,
                       help='Number of columns in the collage (default: 6)')
    parser.add_argument('--collage-target-width', type=int, default=3840,
                       help='Target total collage width in pixels (default: 3840)')
    parser.add_argument('--frame-index', type=int, default=0,
                       help='Zero-based index of frame to extract from each video (default: 0 = first frame)')
    parser.add_argument('--align-by-timecode', action='store_true',
                       help='If set, compute per-camera frame offsets from strict tmcd timecode and add to --frame-index')
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


def annotate_combined_image(image: Image.Image, label_text: str) -> Image.Image:
    """Annotate combined image with a thick right yellow bar and large filename label."""
    draw = ImageDraw.Draw(image)
    height = image.height
    width = image.width
    bar_width = max(8, height // 140)
    # Right-side yellow bar
    draw.rectangle([width - bar_width, 0, width - 1, height - 1], fill='yellow')
    try:
        font_size = max(28, height // 16)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    # Keep label at top-left for readability
    text_x, text_y = 10, 10
    outline_width = max(2, height // 240)
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((text_x + dx, text_y + dy), label_text, font=font, fill='black')
    draw.text((text_x, text_y), label_text, font=font, fill='white')
    return image


def calculate_optimal_grid(num_images: int, img_width: int, img_height: int) -> Tuple[int, int]:
    """Calculate grid layout with fixed 12 columns per row."""
    cols = 12  # Fixed number of columns
    rows = math.ceil(num_images / cols)
    
    return cols, rows


def create_collage_from_paths(
    image_paths: List[str],
    output_path: str,
    scale_factor: float = 1.0,
    verbose: bool = False,
    cols_override: Optional[int] = None,
    draw_separators: bool = True,
    separator_width: int = 2,
    separator_color: str = 'gray',
) -> None:
    """Create a collage by streaming images from disk to reduce memory usage."""
    if not image_paths:
        print("No images to create collage")
        return

    first_img = Image.open(image_paths[0]).convert('RGB')
    img_width, img_height = first_img.size
    if scale_factor != 1.0:
        img_width = int(img_width * scale_factor)
        img_height = int(img_height * scale_factor)

    if cols_override is not None and cols_override > 0:
        cols = cols_override
        rows = math.ceil(len(image_paths) / cols)
    else:
        cols, rows = calculate_optimal_grid(len(image_paths), img_width, img_height)

    if verbose:
        print(f"Creating collage with {cols}×{rows} grid")
        print(f"Image size: {img_width}×{img_height}")
        print(f"Total images: {len(image_paths)}")

    collage_width = cols * img_width
    collage_height = rows * img_height
    collage = Image.new('RGB', (collage_width, collage_height), 'black')

    first_paste = first_img.resize((img_width, img_height), Image.Resampling.LANCZOS) if scale_factor != 1.0 else first_img
    collage.paste(first_paste, (0, 0))
    first_paste = None
    first_img = None

    for i, path in enumerate(tqdm(image_paths[1:], desc="Creating collage", disable=not verbose), start=1):
        row = i // cols
        col = i % cols
        x = col * img_width
        y = row * img_height
        try:
            img = Image.open(path).convert('RGB')
            if scale_factor != 1.0:
                img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
            collage.paste(img, (x, y))
        except Exception as exc:
            print(f"Warning: skipping '{path}' due to error: {exc}")

    # Separators disabled for speed; PNGs are already annotated
    if draw_separators and cols > 1:
        pass

    if verbose:
        print(f"Saving collage: {collage_width}×{collage_height} pixels")
    collage.save(output_path, 'JPEG', quality=95, optimize=True)
    if verbose:
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Collage saved: {output_path} ({file_size:.1f} MB)")

def create_collage(
    images_data: List[Tuple[Image.Image, str]],
    output_path: str,
    scale_factor: float = 1.0,
    verbose: bool = False,
    cols_override: Optional[int] = None,
    draw_separators: bool = True,
    separator_width: int = 2,
    separator_color: str = 'gray',
) -> None:
    """Create a collage from images with labels.

    - cols_override: if provided, fixes number of columns; otherwise auto layout
    - draw_separators: draw vertical lines between columns across the collage
    """
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
    
    # Calculate grid
    if cols_override is not None and cols_override > 0:
        cols = cols_override
        rows = math.ceil(len(images_data) / cols)
    else:
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
    
    # Optional separators between columns
    if draw_separators and cols > 1:
        draw = ImageDraw.Draw(collage)
        for c in range(1, cols):
            x = c * img_width
            # Draw a vertical rectangle as separator
            draw.rectangle(
                [x - max(1, separator_width // 2), 0, x + (separator_width // 2), collage_height - 1],
                fill=separator_color,
            )
    
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


def _ffprobe_json(path: str) -> Optional[dict]:
    """Run ffprobe and return parsed JSON, or None on error."""
    try:
        proc = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-print_format', 'json',
                '-show_format', '-show_streams',
                '-show_entries',
                'stream=index,codec_type,codec_tag_string,avg_frame_rate,r_frame_rate,tags,disposition:format=tags'
            , path],
            capture_output=True, text=True, check=True
        )
        return json.loads(proc.stdout)
    except Exception:
        return None


def _parse_fraction(fr: Optional[str]) -> Optional[float]:
    """Parse fraction like '60/1' to float; return None if invalid."""
    if not fr or fr == '0/0':
        return None
    try:
        num_str, den_str = fr.split('/')
        num = float(num_str)
        den = float(den_str)
        if den == 0:
            return None
        return num / den
    except Exception:
        return None


def _extract_strict_timecode(info: dict) -> Optional[str]:
    """Extract timecode strictly from tmcd stream tag 'timecode'."""
    for s in info.get('streams', []):
        if s.get('codec_type') == 'data' and s.get('codec_tag_string') == 'tmcd':
            tags = s.get('tags') or {}
            tc = tags.get('timecode')
            if tc:
                return tc
    return None


def _extract_video_fps(info: dict) -> Optional[float]:
    """Extract nominal fps from video stream."""
    for s in info.get('streams', []):
        if s.get('codec_type') == 'video':
            return _parse_fraction(s.get('avg_frame_rate')) or _parse_fraction(s.get('r_frame_rate'))
    return None


def _timecode_to_frame_index(tc: str, fps: float) -> int:
    """Convert HH:MM:SS:FF to absolute frame index since midnight (non-drop)."""
    parts = tc.split(':')
    if len(parts) != 4:
        return 0
    hh, mm, ss, ff = [int(x) for x in parts]
    total_seconds = hh * 3600 + mm * 60 + ss
    return int(round(total_seconds * fps + ff))


def compute_timecode_offsets_left_front_right(paths_lfr: List[str], verbose: bool = False) -> Optional[List[int]]:
    """Compute per-camera offsets [left, front, right] based on strict tmcd timecodes.

    Returns list of non-negative frame offsets relative to the latest start, or None if unavailable.
    """
    if len(paths_lfr) != 3:
        return None

    infos: List[Optional[dict]] = [_ffprobe_json(p) for p in paths_lfr]
    if any(info is None for info in infos):
        print("Warning: ffprobe failed for one or more files; cannot align by timecode")
        return None

    # Extract strict timecodes and FPS
    timecodes: List[Optional[str]] = []
    for idx, info in enumerate(infos):
        tc = _extract_strict_timecode(info)  # strictly from tmcd
        if not tc:
            print("Warning: strict timecode (tmcd/timecode) not found; skipping timecode alignment")
            return None
        timecodes.append(tc)

    # FPS check: warn if any video fps != 60
    fps_values: List[float] = []
    for info in infos:
        fps = _extract_video_fps(info) or 60.0
        fps_values.append(fps)
    if any(abs(fps - 60.0) > 0.1 for fps in fps_values):
        print(f"Warning: non-60 fps detected in triplet: {[f'{fps:.2f}' for fps in fps_values]}")

    # Use 60 fps for timecode frame math as per requirement
    nominal_fps = 60.0
    start_frames = [_timecode_to_frame_index(tc, nominal_fps) for tc in timecodes]  # left, front, right
    latest = max(start_frames)
    offsets = [latest - sf for sf in start_frames]
    if verbose:
        print(f"Timecode offsets (frames) -> left: {offsets[0]}, front: {offsets[1]}, right: {offsets[2]}")
    return offsets


def get_video_frame_count(video_path: str) -> Optional[int]:
    """Return total frame count using OpenCV, or None if unavailable."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception:
        count = None
    finally:
        cap.release()
    return count


def main():
    """Main function."""
    args = parse_arguments()
    
    # Convert input directory to absolute path
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Create grouped output directory structure
    input_name = os.path.basename(input_dir.rstrip(os.sep))
    output_base = os.path.abspath(args.output)
    index_key = f"{args.frame_index}_tc" if args.align_by_timecode else f"{args.frame_index}"
    frames_dir = os.path.join(output_base, 'frames', input_name, index_key)
    collages_dir = os.path.join(output_base, 'collages')
    logs_dir = os.path.join(output_base, 'logs')
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(collages_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    if args.verbose:
        print(f"Input directory: {input_dir}")
        print(f"Frames directory: {frames_dir}")
        print(f"Collages directory: {collages_dir}")
        print(f"Logs directory: {logs_dir}")
    
    # Find all video folders
    video_folders = find_strict_triplets(input_dir)
    
    if not video_folders:
        print("No folders with exactly 3 video files found")
        return
    
    if args.verbose:
        print(f"Found {len(video_folders)} video folders to process")
    
    # Store labels for images to include in collage (avoid holding images in memory)
    all_labels: List[str] = []
    # Stats for timecode alignment issues
    num_tc_issue_triplets = 0
    tc_issue_log_lines: List[str] = []
    
    # Process each folder
    for folder_path, label in tqdm(video_folders, desc="Processing video folders"):
        output_filename = f"{label}.png"
        output_path = os.path.join(frames_dir, output_filename)

        # If output already exists, skip processing and remember label for collage
        if os.path.exists(output_path):
            if args.verbose:
                print(f"Skipping existing: {output_path}")
            if not args.no_collage:
                all_labels.append(label)
            continue

        # Compute timecode alignment offsets if requested
        effective_frame_index = args.frame_index
        offsets = None
        if args.align_by_timecode:
            # folder_path is a list [left, front, right]
            paths_lfr = folder_path
            # We need offsets in order [left, front, right], then map to per-path addition
            offsets = compute_timecode_offsets_left_front_right(paths_lfr, verbose=args.verbose)
            if offsets is None:
                print("Warning: could not compute timecode offsets; proceeding without alignment")
            else:
                if args.verbose:
                    print(f"Applying timecode alignment offsets (frames): left={offsets[0]}, front={offsets[1]}, right={offsets[2]}")

        # Process the folder normally, with optional per-camera offsets
        processed_images: List[Image.Image] = []
        per_camera_paths = folder_path  # [left, front, right]
        per_camera_labels = ['left', 'front', 'right']
        triplet_had_issue = False
        per_cam_issue_details: List[str] = []
        for cam_idx, video_path in enumerate(per_camera_paths):
            per_cam_index = args.frame_index
            if args.align_by_timecode and offsets is not None:
                per_cam_index += offsets[cam_idx]

            # Diagnostics: check frame count and clamp index if needed
            total_frames = get_video_frame_count(video_path)
            if total_frames is not None:
                if per_cam_index >= total_frames:
                    triplet_had_issue = True
                    per_cam_issue_details.append(
                        f"{per_camera_labels[cam_idx]}: requested={per_cam_index} total={total_frames} -> clamp={total_frames-1}"
                    )
                    if args.verbose:
                        print(f"Warning: requested frame {per_cam_index} >= total_frames {total_frames} for {per_camera_labels[cam_idx]} ({os.path.basename(video_path)}); clamping to {total_frames-1}")
                    per_cam_index = max(0, total_frames - 1)
                elif per_cam_index < 0:
                    triplet_had_issue = True
                    per_cam_issue_details.append(
                        f"{per_camera_labels[cam_idx]}: requested={per_cam_index} total={total_frames} -> clamp=0"
                    )
                    if args.verbose:
                        print(f"Warning: requested frame {per_cam_index} < 0 for {per_camera_labels[cam_idx]} ({os.path.basename(video_path)}); clamping to 0")
                    per_cam_index = 0

            frame = extract_frame(video_path, per_cam_index)
            if frame is None:
                print(f"Error: failed to extract frame for {per_camera_labels[cam_idx]} at index {per_cam_index}")
                triplet_had_issue = True
                per_cam_issue_details.append(
                    f"{per_camera_labels[cam_idx]}: failed_extract at index={per_cam_index}"
                )
                processed_images = []
                break
            processed_image = process_frame(frame, args.crop_fraction, args.line_position)
            processed_images.append(processed_image)

        combined_image = combine_images(processed_images) if processed_images else None
        if combined_image is not None and not triplet_had_issue:
            # Annotate with label (filename) and right yellow bar for readability
            combined_image = annotate_combined_image(combined_image, label)

        # Log timecode alignment issues per-triplet (front path + offsets)
        if args.align_by_timecode and triplet_had_issue:
            num_tc_issue_triplets += 1
            front_path = per_camera_paths[1] if len(per_camera_paths) > 1 else 'unknown'
            offsets_repr = 'N/A'
            if offsets is not None and len(offsets) == 3:
                offsets_repr = f"left={offsets[0]},front={offsets[1]},right={offsets[2]}"
            tc_issue_log_lines.append(
                f"label={label}\tfront={front_path}\toffsets={offsets_repr}\tissues={' | '.join(per_cam_issue_details) if per_cam_issue_details else 'none'}"
        )
        
        if combined_image is not None and not triplet_had_issue:
            # Save the combined image
            combined_image.save(output_path)
            
            # Store for collage (by default both are enabled unless --no-collage is specified)
            if not args.no_collage:
                all_labels.append(label)
            
            if args.verbose:
                print(f"Saved: {output_path}")
        else:
            print(f"Error processing folder (skipped saving due to issues): {folder_path}")
    
    print(f"Processing complete. Results saved to frames: {frames_dir}")
    
    # Summary for timecode alignment issues
    if args.align_by_timecode:
        total = len(video_folders)
        pct = (num_tc_issue_triplets / total * 100.0) if total else 0.0
        print(f"Timecode alignment issues: {num_tc_issue_triplets}/{total} ({pct:.1f}%)")
        if num_tc_issue_triplets > 0:
            log_name = f"{input_name}_tc_alignment_issues_idx{args.frame_index}.log"
            log_path = os.path.join(logs_dir, log_name)
            try:
                with open(log_path, 'w', encoding='utf-8') as lf:
                    for line in tc_issue_log_lines:
                        lf.write(line + "\n")
                print(f"Details saved to: {log_path}")
            except Exception as exc:
                print(f"Warning: could not write issues log '{log_path}': {exc}")
    
    # Create a single resized collage
    if not args.no_collage and all_labels:
        print(f"\nCreating collages from {len(all_labels)} images...")
        
        # Target: args.collage-cols columns, target width: args.collage-target-width
        # Determine original width by opening only the first saved PNG
        first_path = os.path.join(frames_dir, f"{all_labels[0]}.png")
        try:
            with Image.open(first_path) as _fi:
                original_img_width = _fi.width
        except Exception:
            # Fallback to a sane default
            original_img_width = 640
        cols = max(1, int(args.collage_cols))
        rows = math.ceil(len(all_labels) / cols)
        target_width = max(320, int(args.collage_target_width))
        scale_factor = target_width / (cols * original_img_width)
                
        # Output filename includes only frame index
        collage_name = f"{input_name}_collage_idx{args.frame_index}.jpg"
        collage_path = os.path.join(collages_dir, collage_name)
        print(f"Creating resized collage {cols}×{rows} to width {target_width} px (scale {scale_factor:.3f})...")
        # Stream from disk to reduce memory usage (use saved PNG paths)
        image_paths = [os.path.join(frames_dir, f"{lb}.png") for lb in all_labels]
        create_collage_from_paths(
            image_paths,
            collage_path,
            scale_factor=scale_factor,
            verbose=args.verbose,
            cols_override=cols,
        )
    elif not args.no_collage:
        print("No images available for collage creation. Ensure per-triplet PNGs exist.")


if __name__ == "__main__":
    try:
        main()
        print("Completed successfully.")
    except Exception as exc:
        # Ensure any unexpected exception is visible in debug console
        print(f"Fatal error: {exc}")
        raise