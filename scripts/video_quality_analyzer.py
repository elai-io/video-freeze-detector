import os
import cv2
import numpy as np
import argparse
from typing import Tuple, List
from tqdm import tqdm
from numpy.typing import NDArray


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


def compute_brightness(frame: NDArray) -> float:
    """
    Compute average brightness (mean pixel value in grayscale)
    
    Args:
        frame: Input frame (BGR)
        
    Returns:
        Average brightness value (0-255, higher = brighter)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Return mean pixel value
    return float(np.mean(gray))


def compute_contrast(frame: NDArray) -> float:
    """
    Compute contrast (standard deviation of pixel values)
    
    Args:
        frame: Input frame (BGR)
        
    Returns:
        Contrast value (higher = more contrast)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Return standard deviation
    return float(np.std(gray))


def compute_saturation(frame: NDArray) -> float:
    """
    Compute average saturation (mean S-channel value in HSV)
    
    Args:
        frame: Input frame (BGR)
        
    Returns:
        Average saturation value (0-255, higher = more saturated)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Extract S-channel (saturation)
    saturation = hsv[:, :, 1]
    
    # Return mean saturation
    return float(np.mean(saturation))


def compute_color_balance(frame: NDArray) -> float:
    """
    Compute color balance deviation from neutral (1:1:1 ratio)
    
    Args:
        frame: Input frame (BGR)
        
    Returns:
        Color balance deviation (0 = neutral, higher = more deviation)
    """
    # Split into BGR channels
    b, g, r = cv2.split(frame)
    
    # Compute mean values
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    
    # Compute deviation from neutral (1:1:1 ratio)
    total = b_mean + g_mean + r_mean
    if total == 0:
        return 0.0
    
    # Normalize to get ratios
    b_ratio = b_mean / total
    g_ratio = g_mean / total
    r_ratio = r_mean / total
    
    # Compute deviation from ideal 1/3 ratio
    ideal_ratio = 1.0 / 3.0
    deviation = np.sqrt((b_ratio - ideal_ratio)**2 + (g_ratio - ideal_ratio)**2 + (r_ratio - ideal_ratio)**2)
    
    return float(deviation)


def compute_noise_level(frame: NDArray) -> float:
    """
    Compute noise level using local variance method
    
    Args:
        frame: Input frame (BGR)
        
    Returns:
        Noise level estimate (higher = more noise)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to get "smooth" version
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute difference (noise = original - smooth)
    noise = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    
    # Return mean noise level
    return float(np.mean(noise))


def analyze_video_quality(video_path: str, sample_frames: int = None) -> Tuple[float, float, float, float, float, float]:
    """
    Analyze quality metrics for a single video file
    
    Args:
        video_path: Path to video file
        sample_frames: Number of frames to sample for analysis (None = all frames)
        
    Returns:
        Tuple of (laplacian_variance, tenengrad_variance, brightness, contrast, saturation, color_balance, noise_level)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
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
    brightness_values = []
    contrast_values = []
    saturation_values = []
    color_balance_values = []
    noise_values = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Compute all quality metrics
        laplacian_var = compute_laplacian_variance(frame)
        tenengrad_var = compute_tenengrad_variance(frame)
        brightness = compute_brightness(frame)
        contrast = compute_contrast(frame)
        saturation = compute_saturation(frame)
        color_balance = compute_color_balance(frame)
        noise_level = compute_noise_level(frame)
        
        laplacian_values.append(laplacian_var)
        tenengrad_values.append(tenengrad_var)
        brightness_values.append(brightness)
        contrast_values.append(contrast)
        saturation_values.append(saturation)
        color_balance_values.append(color_balance)
        noise_values.append(noise_level)
    
    cap.release()
    
    # Return average values
    avg_laplacian = float(np.mean(laplacian_values)) if laplacian_values else 0.0
    avg_tenengrad = float(np.mean(tenengrad_values)) if tenengrad_values else 0.0
    avg_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
    avg_contrast = float(np.mean(contrast_values)) if contrast_values else 0.0
    avg_saturation = float(np.mean(saturation_values)) if saturation_values else 0.0
    avg_color_balance = float(np.mean(color_balance_values)) if color_balance_values else 0.0
    avg_noise = float(np.mean(noise_values)) if noise_values else 0.0
    
    return avg_laplacian, avg_tenengrad, avg_brightness, avg_contrast, avg_saturation, avg_color_balance, avg_noise


def main():
    parser = argparse.ArgumentParser(description='Analyze video quality metrics for all video files in a directory')
    parser.add_argument('input_dir', help='Directory containing video files')
    parser.add_argument('--sample-frames', type=int, default=None,
                       help='Number of frames to sample per video (overrides --mode)')
    parser.add_argument('--mode', choices=['fast', 'normal', 'full'], default='fast',
                       help='Analysis mode: fast (10 frames), normal (100 frames), full (all frames) (default: fast)')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed metrics (brightness, contrast, saturation, color balance)')
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
    
    print("📹 VIDEO QUALITY ANALYZER")
    print("=" * 100)
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
        
        laplacian, tenengrad, brightness, contrast, saturation, color_balance, noise = analyze_video_quality(video_path, sample_frames)
        results.append((filename, laplacian, tenengrad, brightness, contrast, saturation, color_balance, noise))
        
        if args.verbose:
            if args.detailed:
                print(f"  Laplacian: {laplacian:.1f}, Tenengrad: {tenengrad:.1f}, Noise: {noise:.1f}, Brightness: {brightness:.1f}, Contrast: {contrast:.1f}, Saturation: {saturation:.1f}, Color Balance: {color_balance:.3f}")
            else:
                print(f"  Laplacian: {laplacian:.1f}, Tenengrad: {tenengrad:.1f}, Noise: {noise:.1f}")
    
    # Calculate dynamic table width based on longest filename
    max_filename_length = max(len(filename) for filename, *_ in results)
    filename_width = max_filename_length + 3  # Add some padding
    
    if args.detailed:
        # Full table with all metrics
        total_width = filename_width + 82  # 82 = sum of all metric column widths
        
        print("\n" + "=" * total_width)
        print("QUALITY ANALYSIS RESULTS (DETAILED)")
        print("=" * total_width)
        print(f"{'Filename':<{filename_width}} {'Sharpness':<10} {'Focus':<10} {'Noise':<10} {'Brightness':<10} {'Contrast':<10} {'Saturation':<10} {'Color Balance':<12}")
        print(f"{'':<{filename_width}} {'Laplacian ↑':<10} {'Tenengrad ↑':<10} {'Level ~':<10} {'Avg ↑':<10} {'Std ↑':<10} {'HSV-S ↑':<10} {'Deviation ↓':<12}")
        print("-" * total_width)
        
        for filename, laplacian, tenengrad, brightness, contrast, saturation, color_balance, noise in results:
            print(f"{filename:<{filename_width}} {laplacian:<10.1f} {tenengrad:<10.1f} {noise:<10.1f} {brightness:<10.1f} {contrast:<10.1f} {saturation:<10.1f} {color_balance:<12.3f}")
        
        print("=" * total_width)
        print(f"✅ Analyzed {len(results)} video files")
        print("Note: ↑ = higher is better, ↓ = lower is better, ~ = expected to be similar")
        print("Metrics: Sharpness (Laplacian), Focus (Tenengrad), Noise (local variance),")
        print("         Brightness (0-255), Contrast (Std Dev), Saturation (HSV-S), Color Balance (deviation)")
    else:
        # Compact table with only main metrics
        total_width = filename_width + 30  # 30 = sum of main metric column widths (10+10+10)
        
        print("\n" + "=" * total_width)
        print("QUALITY ANALYSIS RESULTS")
        print("=" * total_width)
        print(f"{'Filename':<{filename_width}} {'Sharpness':<10} {'Focus':<10} {'Noise':<10}")
        print(f"{'':<{filename_width}} {'Laplacian ↑':<10} {'Tenengrad ↑':<10} {'Level ~':<10}")
        print("-" * total_width)
        
        for filename, laplacian, tenengrad, brightness, contrast, saturation, color_balance, noise in results:
            print(f"{filename:<{filename_width}} {laplacian:<10.1f} {tenengrad:<10.1f} {noise:<10.1f}")
        
        print("=" * total_width)
        print(f"✅ Analyzed {len(results)} video files")
        print("Note: ↑ = higher is better, ↓ = lower is better, ~ = expected to be similar")
        print("Use --detailed flag to see brightness, contrast, saturation, and color balance metrics")
    
    return 0


if __name__ == '__main__':
    exit(main()) 