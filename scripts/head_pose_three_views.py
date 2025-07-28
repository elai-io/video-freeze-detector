import os
import cv2
import argparse
from typing import Tuple, List, Union
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

try:
    # SixDRepNet is installed via pip (pip install sixdrepnet)
    from sixdrepnet import SixDRepNet
except ImportError as exc:
    raise ImportError('SixDRepNet package not found. Install it with "pip install sixdrepnet"') from exc


def analyze_head_pose(
    video_path: str,
    model: 'SixDRepNet',
    sample_frames: Union[int, None] = None,
    verbose: bool = False,
) -> NDArray[np.float64]:
    """Analyze head pose angles for the given video.

    Args:
        video_path: Path to the input video file.
        model: An initialised SixDRepNet model.
        sample_frames: Number of frames to sample. ``None`` means all frames.
        verbose: If ``True`` prints progress for each processed frame.

    Returns:
        Array containing yaw predictions for the processed frames (in degrees).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video file: {video_path}')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        # Some codecs may not set frame count, fall back to reading fully
        total_frames = None

    # Decide on frame indices to sample
    if sample_frames is None or total_frames is None or sample_frames >= total_frames:
        frame_indices = None  # Process sequentially until the end
    else:
        step = max(total_frames // sample_frames, 1)
        frame_indices = set(range(0, total_frames, step))

    yaw_list: List[float] = []

    frame_id = 0
    progress_bar = tqdm(total=sample_frames if sample_frames else total_frames, disable=not verbose)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_indices is not None and frame_id not in frame_indices:
            frame_id += 1
            continue

        # SixDRepNet returns pitch, yaw, roll
        _pitch, yaw, _roll = model.predict(frame)

        yaw_list.append(float(yaw))

        frame_id += 1
        if progress_bar is not None:
            progress_bar.update(1)

        # Early stop if we reached desired sample_frames
        if sample_frames is not None and len(yaw_list) >= sample_frames:
            break

    cap.release()
    if progress_bar is not None:
        progress_bar.close()

    return np.array(yaw_list, dtype=np.float64)


def compute_mean_ci(angles: NDArray[np.float64], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and (lower, upper) bounds of the confidence interval.

    Uses a normal-distribution approximation (Z-score) as data size is
    typically large (≥30). For small samples, this still provides a reasonable
    estimate without requiring external dependencies.
    """
    n = angles.size
    if n == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(angles))
    # Standard error of the mean
    sem = float(np.std(angles, ddof=1)) / np.sqrt(n)

    # Z value for 95% confidence
    z = 1.96 if confidence == 0.95 else 1.0  # simplistic fallback
    margin = z * sem
    return mean_val, mean_val - margin, mean_val + margin


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Analyze head pose angles for three camera views (left, front, right) using SixDRepNet'
    )
    parser.add_argument('--left', required=True, help='Path to the left-view video')
    parser.add_argument('--front', required=True, help='Path to the frontal-view video')
    parser.add_argument('--right', required=True, help='Path to the right-view video')
    parser.add_argument('--sample-frames', type=int, default=0, help='Number of frames to sample from each video (0 = all frames)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    sample_frames = None if args.sample_frames == 0 else args.sample_frames

    print('HEAD POSE ANALYSIS (6DRepNet)')
    print('=' * 100)

    # Initialise model
    model = SixDRepNet()

    print('\nProcessing videos…')

    # Process cameras separately
    yaw_left = analyze_head_pose(args.left, model, sample_frames, args.verbose)
    yaw_front = analyze_head_pose(args.front, model, sample_frames, args.verbose)
    yaw_right = analyze_head_pose(args.right, model, sample_frames, args.verbose)

    # Align lengths
    n = min(yaw_left.size, yaw_front.size, yaw_right.size)
    yaw_left, yaw_front, yaw_right = yaw_left[:n], yaw_front[:n], yaw_right[:n]

    # Compute camera offsets: delta = front - cam
    delta_left = yaw_front - yaw_left
    delta_right = yaw_front - yaw_right

    mean_left, ci_left_low, ci_left_up = compute_mean_ci(delta_left)
    mean_right, ci_right_low, ci_right_up = compute_mean_ci(delta_right)

    margin_left = mean_left - ci_left_low
    margin_right = mean_right - ci_right_low

    print('\nEstimated camera yaw offsets (relative to frontal):')
    print(f"left: {abs(mean_left):.1f} ± {margin_left:.1f} degrees")
    print(f"right: {abs(mean_right):.1f} ± {margin_right:.1f} degrees")
    print('✅ Done')
    return 0


if __name__ == '__main__':
    exit(main()) 