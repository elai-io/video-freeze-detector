import argparse
import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from collections import defaultdict

import ffmpeg


class VideoInfo(NamedTuple):
    """Container for video file information."""
    path: str
    timecode: str
    duration: float
    fps: float
    width: int
    height: int


def extract_video_info(video_path: str) -> Optional[VideoInfo]:
    """Extract comprehensive video information including timecode, duration, fps, and dimensions.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        VideoInfo object with all video metadata or None if extraction failed
    """
    try:
        # Probe the video file to get stream information
        probe_data = ffmpeg.probe(video_path)
        
        # Find video stream
        video_stream = None
        for stream in probe_data['streams']:
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            print(f'No video stream found in {video_path}', file=sys.stderr)
            return None
        
        # Extract basic video info
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        duration = float(video_stream.get('duration', 0))
        
        # Calculate FPS
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(fps_str)
        
        # Look for timecode in data streams first (tmcd streams)
        timecode = None
        data_streams = [
            stream for stream in probe_data['streams'] 
            if stream.get('codec_tag_string') == 'tmcd'
        ]
        
        for stream in data_streams:
            tags = stream.get('tags', {})
            if 'timecode' in tags:
                timecode = tags['timecode']
                break
        
        # If no timecode found in data streams, check video stream
        if not timecode:
            tags = video_stream.get('tags', {})
            timecode = tags.get('timecode')
        
        if not timecode:
            print(f'No timecode found in {video_path}', file=sys.stderr)
            return None
        
        return VideoInfo(
            path=video_path,
            timecode=timecode,
            duration=duration,
            fps=fps,
            width=width,
            height=height
        )
        
    except ffmpeg.Error as e:
        print(f'FFmpeg error extracting info from {video_path}: {e}', file=sys.stderr)
        return None
    except Exception as e:
        print(f'Unexpected error processing {video_path}: {e}', file=sys.stderr)
        return None


def scan_directory(directory: str, extensions: List[str]) -> List[str]:
    """Scan directory for video files with specified extensions.
    
    Args:
        directory: Directory to scan
        extensions: List of file extensions to look for (e.g., ['.mov', '.mp4'])
        
    Returns:
        List of full paths to video files
    """
    if not os.path.exists(directory):
        print(f'Warning: Directory {directory} does not exist')
        return []
    
    video_files = []
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext.lower()) for ext in extensions):
            video_files.append(os.path.join(directory, file))
    
    return video_files


def timecode_to_seconds(timecode: str, fps: float) -> float:
    """Convert timecode to seconds for precise alignment.
    
    Args:
        timecode: Timecode in HH:MM:SS:FF format
        fps: Frames per second
        
    Returns:
        Time in seconds as float
    """
    try:
        parts = timecode.split(':')
        if len(parts) != 4:
            return 0.0
        
        hours, minutes, seconds, frames = map(int, parts)
        total_seconds = hours * 3600 + minutes * 60 + seconds + frames / fps
        return total_seconds
    except ValueError:
        return 0.0


def calculate_alignment_params(left_info: VideoInfo, frontal_info: VideoInfo, right_info: VideoInfo) -> Tuple[float, float, float, float]:
    """Calculate alignment parameters for three videos.
    
    Args:
        left_info: Left camera video info
        frontal_info: Frontal camera video info  
        right_info: Right camera video info
        
    Returns:
        Tuple of (start_offset_left, start_offset_frontal, start_offset_right, aligned_duration)
    """
    # Convert timecodes to seconds using the actual fps of each video
    left_start = timecode_to_seconds(left_info.timecode, left_info.fps)
    frontal_start = timecode_to_seconds(frontal_info.timecode, frontal_info.fps)
    right_start = timecode_to_seconds(right_info.timecode, right_info.fps)
    
    # Find the latest start time (all videos must start from this point)
    latest_start = max(left_start, frontal_start, right_start)
    
    # Calculate how much to trim from the beginning of each video
    left_trim = latest_start - left_start
    frontal_trim = latest_start - frontal_start
    right_trim = latest_start - right_start
    
    # Calculate remaining duration for each video after trimming the start
    left_remaining = left_info.duration - left_trim
    frontal_remaining = frontal_info.duration - frontal_trim
    right_remaining = right_info.duration - right_trim
    
    # Final duration is the shortest remaining duration
    aligned_duration = min(left_remaining, frontal_remaining, right_remaining)
    
    return left_trim, frontal_trim, right_trim, aligned_duration


def process_video(input_path: str, output_path: str, start_offset: float, duration: float, 
                 proxy_resolution: bool = False, fps: float = 60.0) -> bool:
    """Process a single video with trimming and encoding.
    
    Args:
        input_path: Input video file path
        output_path: Output video file path
        start_offset: Seconds to trim from the start
        duration: Duration of the output video
        proxy_resolution: If True, output at 1280x720, otherwise keep original resolution
        fps: Target fps for output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create ffmpeg input
        input_stream = ffmpeg.input(input_path, ss=start_offset, t=duration)
        
        # Configure video encoding
        video_args = {
            'c:v': 'h264_nvenc',  # NVIDIA hardware encoder
            'preset': 'fast',
            'crf': 18,  # High quality
            'r': fps,  # Set frame rate
        }
        
        # Add scaling if proxy resolution is requested
        if proxy_resolution:
            video_args['vf'] = 'scale=1280:720'
        
        # Configure audio encoding
        audio_args = {
            'c:a': 'aac',
            'b:a': '192k'
        }
        
        # Create output stream
        output_stream = ffmpeg.output(
            input_stream,
            output_path,
            **video_args,
            **audio_args,
            loglevel='error',
        )
        
        # Run ffmpeg
        ffmpeg.run(output_stream, overwrite_output=True, quiet=False)
        return True
        
    except ffmpeg.Error as e:
        print(f'FFmpeg error processing {input_path}: {e}', file=sys.stderr)
        return False
    except Exception as e:
        print(f'Unexpected error processing {input_path}: {e}', file=sys.stderr)
        return False


def create_output_directories(base_dir: str) -> Tuple[str, str, str]:
    """Create output directories for left, frontal, and right cameras.
    
    Args:
        base_dir: Base directory where to create the output folders
        
    Returns:
        Tuple of (left_dir, frontal_dir, right_dir) paths
    """
    left_dir = os.path.join(base_dir, 'left')
    frontal_dir = os.path.join(base_dir, 'frontal')
    right_dir = os.path.join(base_dir, 'right')
    
    for dir_path in [left_dir, frontal_dir, right_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return left_dir, frontal_dir, right_dir


def export_timecodes_csv(left_videos: List[VideoInfo], frontal_videos: List[VideoInfo], 
                        right_videos: List[VideoInfo], output_path: str) -> None:
    """Export all timecodes to CSV sorted by time.
    
    Args:
        left_videos: List of left camera videos
        frontal_videos: List of frontal camera videos  
        right_videos: List of right camera videos
        output_path: Path to save the CSV file
    """
    # Collect all timecodes with their info
    all_entries = []
    
    for video in left_videos:
        time_seconds = timecode_to_seconds(video.timecode, video.fps)
        all_entries.append({
            'time_seconds': time_seconds,
            'timecode': video.timecode,
            'camera': 'left',
            'filename': os.path.basename(video.path),
            'full_path': video.path,
            'duration': video.duration,
            'fps': video.fps,
            'width': video.width,
            'height': video.height
        })
    
    for video in frontal_videos:
        time_seconds = timecode_to_seconds(video.timecode, video.fps)
        all_entries.append({
            'time_seconds': time_seconds,
            'timecode': video.timecode,
            'camera': 'frontal',
            'filename': os.path.basename(video.path),
            'full_path': video.path,
            'duration': video.duration,
            'fps': video.fps,
            'width': video.width,
            'height': video.height
        })
    
    for video in right_videos:
        time_seconds = timecode_to_seconds(video.timecode, video.fps)
        all_entries.append({
            'time_seconds': time_seconds,
            'timecode': video.timecode,
            'camera': 'right',
            'filename': os.path.basename(video.path),
            'full_path': video.path,
            'duration': video.duration,
            'fps': video.fps,
            'width': video.width,
            'height': video.height
        })
    
    # Sort by time (oldest to newest)
    all_entries.sort(key=lambda x: x['time_seconds'])
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timecode', 'time_seconds', 'camera', 'filename', 'full_path', 
                     'duration', 'fps', 'width', 'height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in all_entries:
            writer.writerow(entry)
    
    print(f'Exported {len(all_entries)} timecode entries to: {output_path}')


def export_matched_groups_csv(video_groups: List[Tuple[VideoInfo, VideoInfo, VideoInfo]], 
                             output_path: str) -> None:
    """Export matched video groups to CSV.
    
    Args:
        video_groups: List of matched video triplets
        output_path: Path to save the CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'group_id', 'output_name',
            'left_filename', 'left_timecode', 'left_time_seconds', 'left_duration', 
            'frontal_filename', 'frontal_timecode', 'frontal_time_seconds', 'frontal_duration',
            'right_filename', 'right_timecode', 'right_time_seconds', 'right_duration',
            'max_time_diff', 'aligned_duration'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (left_video, frontal_video, right_video) in enumerate(video_groups, 1):
            left_time = timecode_to_seconds(left_video.timecode, left_video.fps)
            frontal_time = timecode_to_seconds(frontal_video.timecode, frontal_video.fps)
            right_time = timecode_to_seconds(right_video.timecode, right_video.fps)
            
            # Calculate alignment parameters
            left_trim, frontal_trim, right_trim, aligned_duration = calculate_alignment_params(
                left_video, frontal_video, right_video
            )
            
            max_time_diff = max(
                abs(left_time - frontal_time),
                abs(left_time - right_time),
                abs(frontal_time - right_time)
            )
            
            ext = os.path.splitext(left_video.path)[1]
            output_name = f'{i:03d}{ext}'
            
            writer.writerow({
                'group_id': i,
                'output_name': output_name,
                'left_filename': os.path.basename(left_video.path),
                'left_timecode': left_video.timecode,
                'left_time_seconds': left_time,
                'left_duration': left_video.duration,
                'frontal_filename': os.path.basename(frontal_video.path), 
                'frontal_timecode': frontal_video.timecode,
                'frontal_time_seconds': frontal_time,
                'frontal_duration': frontal_video.duration,
                'right_filename': os.path.basename(right_video.path),
                'right_timecode': right_video.timecode,
                'right_time_seconds': right_time,
                'right_duration': right_video.duration,
                'max_time_diff': max_time_diff,
                'aligned_duration': aligned_duration
            })
    
    print(f'Exported {len(video_groups)} matched groups to: {output_path}')


def analyze_coverage_statistics(left_videos: List[VideoInfo], frontal_videos: List[VideoInfo], 
                               right_videos: List[VideoInfo], video_groups: List[Tuple[VideoInfo, VideoInfo, VideoInfo]],
                               tolerance: float) -> None:
    """Analyze and print comprehensive coverage statistics.
    
    Args:
        left_videos: All left camera videos
        frontal_videos: All frontal camera videos
        right_videos: All right camera videos
        video_groups: Successfully matched triplets
        tolerance: Tolerance used for matching
    """
    print(f'\n{"="*60}')
    print('COVERAGE STATISTICS')
    print(f'{"="*60}')
    
    total_videos = len(left_videos) + len(frontal_videos) + len(right_videos)
    
    # Videos used in triplets
    used_left_paths = set(group[0].path for group in video_groups)
    used_frontal_paths = set(group[1].path for group in video_groups)
    used_right_paths = set(group[2].path for group in video_groups)
    
    remaining_left = [v for v in left_videos if v.path not in used_left_paths]
    remaining_frontal = [v for v in frontal_videos if v.path not in used_frontal_paths]
    remaining_right = [v for v in right_videos if v.path not in used_right_paths]
    
    # Find pairs from remaining videos
    pairs = []
    used_in_pairs_left = set()
    used_in_pairs_frontal = set()
    used_in_pairs_right = set()
    
    # Left-Frontal pairs
    for left_vid in remaining_left:
        if left_vid.path in used_in_pairs_left:
            continue
        left_time = timecode_to_seconds(left_vid.timecode, left_vid.fps)
        
        best_frontal = None
        best_diff = float('inf')
        
        for frontal_vid in remaining_frontal:
            if frontal_vid.path in used_in_pairs_frontal:
                continue
            frontal_time = timecode_to_seconds(frontal_vid.timecode, frontal_vid.fps)
            diff = abs(left_time - frontal_time)
            
            if diff < tolerance and diff < best_diff:
                best_diff = diff
                best_frontal = frontal_vid
        
        if best_frontal:
            pairs.append(('left-frontal', left_vid, best_frontal, best_diff))
            used_in_pairs_left.add(left_vid.path)
            used_in_pairs_frontal.add(best_frontal.path)
    
    # Left-Right pairs
    for left_vid in remaining_left:
        if left_vid.path in used_in_pairs_left:
            continue
        left_time = timecode_to_seconds(left_vid.timecode, left_vid.fps)
        
        best_right = None
        best_diff = float('inf')
        
        for right_vid in remaining_right:
            if right_vid.path in used_in_pairs_right:
                continue
            right_time = timecode_to_seconds(right_vid.timecode, right_vid.fps)
            diff = abs(left_time - right_time)
            
            if diff < tolerance and diff < best_diff:
                best_diff = diff
                best_right = right_vid
        
        if best_right:
            pairs.append(('left-right', left_vid, best_right, best_diff))
            used_in_pairs_left.add(left_vid.path)
            used_in_pairs_right.add(best_right.path)
    
    # Frontal-Right pairs
    for frontal_vid in remaining_frontal:
        if frontal_vid.path in used_in_pairs_frontal:
            continue
        frontal_time = timecode_to_seconds(frontal_vid.timecode, frontal_vid.fps)
        
        best_right = None
        best_diff = float('inf')
        
        for right_vid in remaining_right:
            if right_vid.path in used_in_pairs_right:
                continue
            right_time = timecode_to_seconds(right_vid.timecode, right_vid.fps)
            diff = abs(frontal_time - right_time)
            
            if diff < tolerance and diff < best_diff:
                best_diff = diff
                best_right = right_vid
        
        if best_right:
            pairs.append(('frontal-right', frontal_vid, best_right, best_diff))
            used_in_pairs_frontal.add(frontal_vid.path)
            used_in_pairs_right.add(best_right.path)
    
    # Count singles (videos not used in triplets or pairs)
    all_used_paths = (used_left_paths | used_frontal_paths | used_right_paths | 
                     used_in_pairs_left | used_in_pairs_frontal | used_in_pairs_right)
    
    singles_left = [v for v in left_videos if v.path not in all_used_paths]
    singles_frontal = [v for v in frontal_videos if v.path not in all_used_paths]  
    singles_right = [v for v in right_videos if v.path not in all_used_paths]
    
    total_singles = len(singles_left) + len(singles_frontal) + len(singles_right)
    
    # Calculate statistics
    triplet_videos = len(video_groups) * 3
    pair_videos = len(pairs) * 2
    
    print(f'Total videos found: {total_videos}')
    print(f'  Left: {len(left_videos)}, Frontal: {len(frontal_videos)}, Right: {len(right_videos)}')
    print()
    
    print(f'TRIPLETS (3-camera groups): {len(video_groups)} groups')
    print(f'  Videos in triplets: {triplet_videos} ({triplet_videos/total_videos*100:.1f}%)')
    print()
    
    print(f'PAIRS (2-camera groups): {len(pairs)} groups')
    print(f'  Videos in pairs: {pair_videos} ({pair_videos/total_videos*100:.1f}%)')
    if pairs:
        pair_breakdown = {}
        for pair_type, _, _, _ in pairs:
            pair_breakdown[pair_type] = pair_breakdown.get(pair_type, 0) + 1
        for pair_type, count in pair_breakdown.items():
            print(f'    {pair_type}: {count} pairs')
    print()
    
    print(f'SINGLES (1-camera only): {total_singles} videos ({total_singles/total_videos*100:.1f}%)')
    print(f'  Left only: {len(singles_left)}, Frontal only: {len(singles_frontal)}, Right only: {len(singles_right)}')
    print()
    
    print('COVERAGE SUMMARY:')
    print(f'  Complete (3-cam): {len(video_groups)} scenes ({len(video_groups)/(len(video_groups) + len(pairs) + total_singles)*100:.1f}% of scenes)')
    print(f'  Partial (2-cam):  {len(pairs)} scenes ({len(pairs)/(len(video_groups) + len(pairs) + total_singles)*100:.1f}% of scenes)') 
    print(f'  Isolated (1-cam): {total_singles} scenes ({total_singles/(len(video_groups) + len(pairs) + total_singles)*100:.1f}% of scenes)')
    
    videos_with_coverage = triplet_videos + pair_videos + total_singles
    print(f'  Total scenes: {len(video_groups) + len(pairs) + total_singles}')
    print(f'  Video utilization: {videos_with_coverage}/{total_videos} ({videos_with_coverage/total_videos*100:.1f}%)')


def export_coverage_analysis_csv(left_videos: List[VideoInfo], frontal_videos: List[VideoInfo], 
                                right_videos: List[VideoInfo], video_groups: List[Tuple[VideoInfo, VideoInfo, VideoInfo]],
                                tolerance: float, output_path: str) -> None:
    """Export coverage analysis including pairs and singles to CSV.
    
    Args:
        left_videos: All left camera videos
        frontal_videos: All frontal camera videos  
        right_videos: All right camera videos
        video_groups: Successfully matched triplets
        tolerance: Tolerance used for matching
        output_path: Path to save the CSV file
    """
    # Reuse the logic from analyze_coverage_statistics to find pairs and singles
    used_left_paths = set(group[0].path for group in video_groups)
    used_frontal_paths = set(group[1].path for group in video_groups)
    used_right_paths = set(group[2].path for group in video_groups)
    
    remaining_left = [v for v in left_videos if v.path not in used_left_paths]
    remaining_frontal = [v for v in frontal_videos if v.path not in used_frontal_paths]
    remaining_right = [v for v in right_videos if v.path not in used_right_paths]
    
    # Find pairs (simplified version of the above logic)
    pairs = []
    used_in_pairs = set()
    
    # Find all possible pairs
    for left_vid in remaining_left:
        if left_vid.path in used_in_pairs:
            continue
        left_time = timecode_to_seconds(left_vid.timecode, left_vid.fps)
        
        # Check frontal matches
        for frontal_vid in remaining_frontal:
            if frontal_vid.path in used_in_pairs:
                continue
            frontal_time = timecode_to_seconds(frontal_vid.timecode, frontal_vid.fps)
            if abs(left_time - frontal_time) < tolerance:
                pairs.append(('left-frontal', left_vid, frontal_vid))
                used_in_pairs.add(left_vid.path)
                used_in_pairs.add(frontal_vid.path)
                break
    
    # Find singles
    all_used_paths = (used_left_paths | used_frontal_paths | used_right_paths | used_in_pairs)
    singles = []
    
    for vid in left_videos + frontal_videos + right_videos:
        if vid.path not in all_used_paths:
            camera = 'left' if vid in left_videos else 'frontal' if vid in frontal_videos else 'right'
            singles.append((camera, vid))
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['coverage_type', 'group_id', 'camera_1', 'file_1', 'timecode_1', 'camera_2', 'file_2', 'timecode_2', 'camera_3', 'file_3', 'timecode_3']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write triplets
        for i, (left, frontal, right) in enumerate(video_groups, 1):
            writer.writerow({
                'coverage_type': 'triplet',
                'group_id': i,
                'camera_1': 'left',
                'file_1': os.path.basename(left.path),
                'timecode_1': left.timecode,
                'camera_2': 'frontal', 
                'file_2': os.path.basename(frontal.path),
                'timecode_2': frontal.timecode,
                'camera_3': 'right',
                'file_3': os.path.basename(right.path),
                'timecode_3': right.timecode
            })
        
        # Write pairs
        for i, (pair_type, vid1, vid2) in enumerate(pairs, 1):
            cameras = pair_type.split('-')
            writer.writerow({
                'coverage_type': 'pair',
                'group_id': i,
                'camera_1': cameras[0],
                'file_1': os.path.basename(vid1.path),
                'timecode_1': vid1.timecode,
                'camera_2': cameras[1],
                'file_2': os.path.basename(vid2.path), 
                'timecode_2': vid2.timecode,
                'camera_3': '',
                'file_3': '',
                'timecode_3': ''
            })
        
        # Write singles
        for i, (camera, vid) in enumerate(singles, 1):
            writer.writerow({
                'coverage_type': 'single',
                'group_id': i,
                'camera_1': camera,
                'file_1': os.path.basename(vid.path),
                'timecode_1': vid.timecode,
                'camera_2': '',
                'file_2': '',
                'timecode_2': '',
                'camera_3': '',
                'file_3': '',
                'timecode_3': ''
            })
    
    print(f'Exported coverage analysis to: {output_path}')


def main():
    """Main function to align multi-camera video files by timecode with video processing."""
    parser = argparse.ArgumentParser(
        description='Align multi-camera video files by timecode with precise trimming and encoding'
    )
    parser.add_argument('left_dir', help='Directory containing left camera view files')
    parser.add_argument('frontal_dir', help='Directory containing frontal camera view files') 
    parser.add_argument('right_dir', help='Directory containing right camera view files')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory for aligned videos (default: current directory)')
    parser.add_argument('--extensions', '-e', nargs='+', default=['.MOV', '.mp4'], 
                       help='Video file extensions to process (default: .MOV .mp4)')
    parser.add_argument('--proxy', action='store_true', help='Generate 1280x720 proxy videos instead of full resolution')
    parser.add_argument('--fps', type=float, default=60.0, help='Target frame rate for output videos (default: 60.0)')
    parser.add_argument('--tolerance', type=float, default=30.0, help='Maximum time difference in seconds for matching videos (default: 30.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output for matching process')
    
    args = parser.parse_args()
    
    # Scan directories for video files
    print('Scanning directories for video files...')
    left_files = scan_directory(args.left_dir, args.extensions)
    frontal_files = scan_directory(args.frontal_dir, args.extensions)
    right_files = scan_directory(args.right_dir, args.extensions)
    
    print(f'Found {len(left_files)} left files, {len(frontal_files)} frontal files, {len(right_files)} right files')
    
    # Extract video info from all files
    print('Extracting video information...')
    left_videos: List[VideoInfo] = []
    frontal_videos: List[VideoInfo] = []
    right_videos: List[VideoInfo] = []
    
    for file_path in left_files:
        info = extract_video_info(file_path)
        if info:
            left_videos.append(info)
        else:
            print(f'Warning: Could not extract info from left file: {file_path}')
    
    for file_path in frontal_files:
        info = extract_video_info(file_path)
        if info:
            frontal_videos.append(info)
        else:
            print(f'Warning: Could not extract info from frontal file: {file_path}')
    
    for file_path in right_files:
        info = extract_video_info(file_path)
        if info:
            right_videos.append(info)
        else:
            print(f'Warning: Could not extract info from right file: {file_path}')
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export all timecodes to CSV for analysis
    csv_path = os.path.join(args.output_dir, 'all_timecodes.csv')
    export_timecodes_csv(left_videos, frontal_videos, right_videos, csv_path)
    
    # Group videos by proximity of timecodes (allowing for timing differences)
    print('Grouping videos by timecode proximity...')
    print(f'Using tolerance: {args.tolerance} seconds')
    video_groups: List[Tuple[VideoInfo, VideoInfo, VideoInfo]] = []
    
    # Convert all timecodes to seconds for easier comparison
    left_times = [(i, timecode_to_seconds(v.timecode, v.fps), v) for i, v in enumerate(left_videos)]
    frontal_times = [(i, timecode_to_seconds(v.timecode, v.fps), v) for i, v in enumerate(frontal_videos)]
    right_times = [(i, timecode_to_seconds(v.timecode, v.fps), v) for i, v in enumerate(right_videos)]
    
    if args.debug:
        print(f'Left videos: {len(left_times)} files')
        for i, t, v in left_times[:5]:  # Show first 5
            print(f'  {os.path.basename(v.path)}: {v.timecode} ({t:.1f}s)')
        print(f'Frontal videos: {len(frontal_times)} files')
        for i, t, v in frontal_times[:5]:
            print(f'  {os.path.basename(v.path)}: {v.timecode} ({t:.1f}s)')
        print(f'Right videos: {len(right_times)} files')  
        for i, t, v in right_times[:5]:
            print(f'  {os.path.basename(v.path)}: {v.timecode} ({t:.1f}s)')
    
    # Find optimal triplets using a more intelligent approach
    used_left = set()
    used_frontal = set()
    used_right = set()
    
    # Sort videos by time to process in chronological order
    left_times.sort(key=lambda x: x[1])
    
    for left_idx, left_time, left_video in left_times:
        if left_idx in used_left:
            continue
            
        best_triplet = None
        best_total_diff = float('inf')
        
        # Find best frontal match for this left video
        for frontal_idx, frontal_time, frontal_video in frontal_times:
            if frontal_idx in used_frontal:
                continue
                
            frontal_diff = abs(left_time - frontal_time)
            if frontal_diff > args.tolerance:
                continue
                
            # Find best right match for this potential pair
            for right_idx, right_time, right_video in right_times:
                if right_idx in used_right:
                    continue
                    
                right_diff = abs(left_time - right_time)
                if right_diff > args.tolerance:
                    continue
                
                # Also check frontal-right compatibility  
                frontal_right_diff = abs(frontal_time - right_time)
                if frontal_right_diff > args.tolerance:
                    continue
                
                # Calculate total "cost" of this triplet
                total_diff = frontal_diff + right_diff + frontal_right_diff
                
                if total_diff < best_total_diff:
                    best_total_diff = total_diff
                    best_triplet = (left_video, frontal_video, right_video, 
                                  left_idx, frontal_idx, right_idx, 
                                  frontal_diff, right_diff, frontal_right_diff)
        
        # Add the best triplet found
        if best_triplet:
            left_vid, frontal_vid, right_vid, l_idx, f_idx, r_idx, f_diff, r_diff, fr_diff = best_triplet
            video_groups.append((left_vid, frontal_vid, right_vid))
            used_left.add(l_idx)
            used_frontal.add(f_idx)
            used_right.add(r_idx)
            
            if args.debug:
                print(f'Matched: {os.path.basename(left_vid.path)} ({left_vid.timecode}) + '
                      f'{os.path.basename(frontal_vid.path)} ({frontal_vid.timecode}) + '
                      f'{os.path.basename(right_vid.path)} ({right_vid.timecode}) '
                      f'[diffs: L-F={f_diff:.1f}s, L-R={r_diff:.1f}s, F-R={fr_diff:.1f}s]')
    
    if not video_groups:
        print('Error: No video groups found with matching timecodes within tolerance')
        return
    
    print(f'Found {len(video_groups)} video groups with matching timecodes')
    
    # Sort groups by earliest timecode
    video_groups.sort(key=lambda group: min(
        timecode_to_seconds(group[0].timecode, group[0].fps),
        timecode_to_seconds(group[1].timecode, group[1].fps),
        timecode_to_seconds(group[2].timecode, group[2].fps)
    ))
    
    # Export matched groups to CSV
    if video_groups:
        groups_csv_path = os.path.join(args.output_dir, 'matched_groups.csv')
        export_matched_groups_csv(video_groups, groups_csv_path)
    
    # Create output directories
    left_out_dir, frontal_out_dir, right_out_dir = create_output_directories(args.output_dir)
    
    # Process video groups
    print('Processing aligned videos...')
    successful_groups = 0
    
    for i, (left_video, frontal_video, right_video) in enumerate(video_groups, 1):
        print(f'\nProcessing group {i}/{len(video_groups)}:')
        print(f'  Left: {os.path.basename(left_video.path)} ({left_video.timecode})')
        print(f'  Frontal: {os.path.basename(frontal_video.path)} ({frontal_video.timecode})')
        print(f'  Right: {os.path.basename(right_video.path)} ({right_video.timecode})')
        
        # Calculate alignment parameters
        left_trim, frontal_trim, right_trim, duration = calculate_alignment_params(
            left_video, frontal_video, right_video
        )
        
        print(f'  Alignment: left_trim={left_trim:.3f}s, frontal_trim={frontal_trim:.3f}s, right_trim={right_trim:.3f}s, duration={duration:.3f}s')
        
        # Skip if duration is too short
        if duration < 1.0:
            print(f'  Skipping group {i}: aligned duration too short ({duration:.3f}s)')
            continue
        
        # Determine file extension and output names
        ext = os.path.splitext(left_video.path)[1]
        output_name = f'{i:03d}{ext}'
        
        left_output = os.path.join(left_out_dir, output_name)
        frontal_output = os.path.join(frontal_out_dir, output_name)
        right_output = os.path.join(right_out_dir, output_name)
        
        # Process each video
        success_count = 0
        
        # Process left video
        if process_video(left_video.path, left_output, left_trim, duration, args.proxy, args.fps):
            print(f'  ✓ Left video processed successfully')
            success_count += 1
        else:
            print(f'  ✗ Failed to process left video')
        
        # Process frontal video
        if process_video(frontal_video.path, frontal_output, frontal_trim, duration, args.proxy, args.fps):
            print(f'  ✓ Frontal video processed successfully')
            success_count += 1
        else:
            print(f'  ✗ Failed to process frontal video')
        
        # Process right video
        if process_video(right_video.path, right_output, right_trim, duration, args.proxy, args.fps):
            print(f'  ✓ Right video processed successfully')
            success_count += 1
        else:
            print(f'  ✗ Failed to process right video')
        
        if success_count == 3:
            successful_groups += 1
            print(f'  ✓ Group {i} completed successfully')
        else:
            print(f'  ⚠ Group {i} completed with {3-success_count} failures')
    
    # Report unmatched videos with analysis
    used_left_paths = set(group[0].path for group in video_groups)
    used_frontal_paths = set(group[1].path for group in video_groups)
    used_right_paths = set(group[2].path for group in video_groups)
    
    unmatched_left = [v for v in left_videos if v.path not in used_left_paths]
    unmatched_frontal = [v for v in frontal_videos if v.path not in used_frontal_paths]
    unmatched_right = [v for v in right_videos if v.path not in used_right_paths]
    
    if unmatched_left or unmatched_frontal or unmatched_right:
        print(f'\nWarnings - Unmatched videos ({len(unmatched_left)} left, {len(unmatched_frontal)} frontal, {len(unmatched_right)} right):')
        
        if args.debug and len(unmatched_left) < 20:  # Only show details for reasonable numbers
            print('Analyzing why videos were unmatched...')
            
            for left_vid in unmatched_left[:10]:  # Show first 10
                left_time = timecode_to_seconds(left_vid.timecode, left_vid.fps)
                closest_frontal = None
                closest_frontal_diff = float('inf')
                closest_right = None  
                closest_right_diff = float('inf')
                
                for frontal_vid in frontal_videos:
                    frontal_time = timecode_to_seconds(frontal_vid.timecode, frontal_vid.fps)
                    diff = abs(left_time - frontal_time)
                    if diff < closest_frontal_diff:
                        closest_frontal_diff = diff
                        closest_frontal = frontal_vid
                
                for right_vid in right_videos:
                    right_time = timecode_to_seconds(right_vid.timecode, right_vid.fps) 
                    diff = abs(left_time - right_time)
                    if diff < closest_right_diff:
                        closest_right_diff = diff
                        closest_right = right_vid
                
                print(f'  Left: {os.path.basename(left_vid.path)} ({left_vid.timecode})')
                if closest_frontal:
                    status = 'USED' if closest_frontal.path in used_frontal_paths else 'AVAIL'
                    print(f'    Closest frontal: {os.path.basename(closest_frontal.path)} ({closest_frontal.timecode}) - {closest_frontal_diff:.1f}s diff [{status}]')
                if closest_right:
                    status = 'USED' if closest_right.path in used_right_paths else 'AVAIL' 
                    print(f'    Closest right: {os.path.basename(closest_right.path)} ({closest_right.timecode}) - {closest_right_diff:.1f}s diff [{status}]')
        else:
            # Just show count summary
            for video in unmatched_left[:5]:
                print(f'  Left: {os.path.basename(video.path)} ({video.timecode})')
            if len(unmatched_left) > 5:
                print(f'  ... and {len(unmatched_left)-5} more left videos')
                
            for video in unmatched_frontal[:5]: 
                print(f'  Frontal: {os.path.basename(video.path)} ({video.timecode})')
            if len(unmatched_frontal) > 5:
                print(f'  ... and {len(unmatched_frontal)-5} more frontal videos')
                
            for video in unmatched_right[:5]:
                print(f'  Right: {os.path.basename(video.path)} ({video.timecode})')
            if len(unmatched_right) > 5:
                print(f'  ... and {len(unmatched_right)-5} more right videos')
    
    resolution_info = "1280x720 proxy" if args.proxy else "original resolution"
    print(f'\nProcessing completed: {successful_groups}/{len(video_groups)} groups processed successfully')
    print(f'Output format: {resolution_info} @ {args.fps} fps')
    print(f'Output directories:')
    print(f'  {left_out_dir}')
    print(f'  {frontal_out_dir}')
    print(f'  {right_out_dir}')
    
    # Generate comprehensive coverage statistics
    analyze_coverage_statistics(left_videos, frontal_videos, right_videos, video_groups, args.tolerance)
    
    # Export coverage analysis CSV
    coverage_csv_path = os.path.join(args.output_dir, 'coverage_analysis.csv')
    export_coverage_analysis_csv(left_videos, frontal_videos, right_videos, video_groups, args.tolerance, coverage_csv_path)


if __name__ == '__main__':
    main()