import os
import cv2
import numpy as np
import argparse
from typing import List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm


def get_video_files(input_dir: str) -> List[str]:
    """Get list of video files from input directory."""
    video_files = []
    supported_extensions = ('.avi', '.mp4', '.mov', '.mkv')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            video_files.append(os.path.join(input_dir, filename))
    
    return sorted(video_files)


def compute_laplacian_variance(frame: NDArray) -> float:
    """
    Compute Laplacian variance (measure of image sharpness/focus)
    
    Args:
        frame: Input frame (BGR)
        
    Returns:
        Laplacian variance value (higher = sharper)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Return variance of Laplacian
    return float(np.var(laplacian))


def compute_tenengrad_variance(frame: NDArray) -> float:
    """
    Compute Tenengrad variance (another measure of image sharpness)
    
    Args:
        frame: Input frame (BGR)
        
    Returns:
        Tenengrad variance value (higher = sharper)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel filters
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Return variance of gradient magnitude
    return float(np.var(gradient_magnitude))


def analyze_video_quality(video_path: str, sample_frames: int = None) -> Tuple[float, float]:
    """
    Analyze quality metrics for a single video file
    
    Args:
        video_path: Path to video file
        sample_frames: Number of frames to sample for analysis (None = all frames)
        
    Returns:
        Tuple of (laplacian_variance, tenengrad_variance)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0.0, 0.0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generate frame indices to sample
    if sample_frames is None:
        # Use all frames
        frame_indices = list(range(total_frames)) if total_frames > 0 else []
    else:
        # Sample specified number of frames
        actual_sample_frames = min(sample_frames, total_frames)
        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, actual_sample_frames, dtype=int)
        else:
            frame_indices = []
    
    laplacian_values = []
    tenengrad_values = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Compute quality metrics
        laplacian_var = compute_laplacian_variance(frame)
        tenengrad_var = compute_tenengrad_variance(frame)
        
        laplacian_values.append(laplacian_var)
        tenengrad_values.append(tenengrad_var)
    
    cap.release()
    
    # Return average values
    avg_laplacian = float(np.mean(laplacian_values)) if laplacian_values else 0.0
    avg_tenengrad = float(np.mean(tenengrad_values)) if tenengrad_values else 0.0
    
    return avg_laplacian, avg_tenengrad


def main():
    parser = argparse.ArgumentParser(description='Analyze video quality metrics for all video files in a directory')
    parser.add_argument('input_dir', help='Directory containing video files')
    parser.add_argument('--sample-frames', type=int, default=None,
                       help='Number of frames to sample per video (overrides --mode)')
    parser.add_argument('--mode', choices=['fast', 'normal', 'full'], default='fast',
                       help='Analysis mode: fast (10 frames), normal (100 frames), full (all frames) (default: fast)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine sample frames based on mode or explicit setting
    if args.sample_frames is not None:
        sample_frames = args.sample_frames
        mode_description = f"custom ({sample_frames} frames)"
    else:
        mode_map = {
            'fast': 10,
            'normal': 100,
            'full': None
        }
        sample_frames = mode_map[args.mode]
        if sample_frames is None:
            mode_description = f"{args.mode} (all frames)"
        else:
            mode_description = f"{args.mode} ({sample_frames} frames)"
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory {args.input_dir} not found")
        return 1
    
    # Get video files
    video_files = get_video_files(args.input_dir)
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        print("Supported formats: AVI, MP4, MOV, MKV")
        return 1
    
    print(f"📹 VIDEO QUALITY ANALYZER")
    print(f"=" * 100)
    print(f"Directory: {args.input_dir}")
    print(f"Found {len(video_files)} video files")
    print(f"Analysis mode: {mode_description}")
    print("-" * 100)
    
    # Analyze each video file
    results = []
    
    video_iter = video_files
    if len(video_files) > 5:
        video_iter = tqdm(video_files, desc="Analyzing videos", unit="video")
    
    for video_path in video_iter:
        filename = os.path.basename(video_path)
        
        if args.verbose:
            print(f"Analyzing: {filename}")
        
        laplacian, tenengrad = analyze_video_quality(video_path, sample_frames)
        results.append((filename, laplacian, tenengrad))
        
        if args.verbose:
            print(f"  Laplacian: {laplacian:.1f}, Tenengrad: {tenengrad:.1f}")
    
    # Display results
    print("\n" + "=" * 100)
    print("QUALITY ANALYSIS RESULTS")
    print("=" * 100)
    print(f"{'Filename':<70} {'Laplacian ↑':<12} {'Tenengrad ↑':<12}")
    print("-" * 100)
    
    for filename, laplacian, tenengrad in results:
        # Truncate filename if too long
        display_name = filename[:67] + "..." if len(filename) > 70 else filename
        print(f"{display_name:<70} {laplacian:<12.1f} {tenengrad:<12.1f}")
    
    print("=" * 100)
    print(f"✅ Analyzed {len(results)} video files")
    print("Note: Higher values indicate better quality (↑)")
    
    return 0


if __name__ == '__main__':
    exit(main()) 