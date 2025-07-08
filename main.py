import argparse
import os
import sys
from typing import List, Tuple
import json

from modules.video_analyzer import VideoAnalyzer
from modules.freeze_detector_edge import FreezeDetectorEdge
from modules.visualizer_edge import VisualizerEdge


def main():
    parser = argparse.ArgumentParser(description='Video freeze detector with comprehensive analysis')
    parser.add_argument('input_path', help='Path to folder with three video files (AVI/MP4)')
    parser.add_argument('--output', '-o', default='output', help='Output folder for results')
    parser.add_argument('--freeze-threshold', '-t', type=float, default=0.25,
                        help='Freeze detection threshold (default 0.25)')
    parser.add_argument('--visualization-percent', '-p', type=float, default=5.0,
                        help='Percentage of most suspicious frames to visualize (default 5.0%%)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    
    args = parser.parse_args()
    
    # Check input folder
    if not os.path.exists(args.input_path):
        print(f"Error: folder {args.input_path} not found")
        sys.exit(1)
    
    # Create output folder
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)
    
    print(f"🎬 VIDEO FREEZE DETECTOR WITH COMPREHENSIVE ANALYSIS")
    print(f"=" * 70)
    print(f"Analyzing video files in folder: {args.input_path}")
    print(f"Results will be saved to: {output_path}")
    print(f"Freeze detection threshold: {args.freeze_threshold}")
    print(f"Visualization percent: {args.visualization_percent}%")
    print("-" * 70)
    
    try:
        # Stage 1: Load and validate videos
        print("Stage 1: Loading and checking video files...")
        analyzer = VideoAnalyzer(args.input_path, verbose=args.verbose)
        video_files = analyzer.load_videos()
        
        if not analyzer.validate_synchronization():
            print("Warning: video files are not synchronized!")
            print("Frame counts:")
            for i, (file_path, frame_count) in enumerate(zip(video_files, analyzer.frame_counts)):
                print(f"  Camera {i+1} ({os.path.basename(file_path)}): {frame_count} frames")
            print()
        
        # Stage 2: Frame analysis
        print("Stage 2: Analyzing frame differences...")
        print("  - Computing edge-based frame differences...")
        edge_frame_differences = analyzer.compute_edge_frame_differences()
        
        # Stage 3: Freeze detection with new algorithm
        print("Stage 3: Running comprehensive freeze analysis...")
        detector = FreezeDetectorEdge(edge_frame_differences, 
                                    freeze_threshold=args.freeze_threshold)
        
        # Get freeze candidates for visualization (top N% most suspicious)
        freeze_candidates = detector.detect_freezes_for_visualization(args.visualization_percent)
        print(f"Found {len(freeze_candidates)} frames for visualization (top {args.visualization_percent}%)")
        
        # Stage 4: Save data to CSV/Excel
        print("Stage 4: Saving analysis data...")
        csv_path = detector.save_to_csv(output_path)
        xlsx_path = detector.save_to_xlsx(output_path)
        
        # Stage 5: Create visualization images
        print("Stage 5: Creating visualization images...")
        print("  - Creating previous/current frame pairs...")
        print("  - Creating edge difference images...")
        
        visualizer = VisualizerEdge(analyzer, output_path)
        visualizer.create_freeze_images(freeze_candidates)
        
        # Stage 6: Generate statistics and reporting
        print("Stage 6: Generating analysis report...")
        stats = detector.generate_statistics()
        
        # Save JSON report
        report_path = os.path.join(output_path, 'freeze_analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json_results = {
                'analysis_parameters': {
                    'input_path': args.input_path,
                    'freeze_threshold': args.freeze_threshold,
                    'total_videos': len(video_files),
                    'video_files': [os.path.basename(f) for f in video_files]
                },
                'statistics': stats,
                'camera_statistics': detector.camera_statistics,
                'suspicious_sequences': detector.suspicious_sequences,
                'longest_sequences_per_camera': detector.get_top_longest_sequences(top_n=20, min_length=2),
                'file_paths': {
                    'csv_path': csv_path,
                    'xlsx_path': xlsx_path,
                    'json_report': report_path
                }
            }
            json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Display results
        print_results(stats, video_files, freeze_candidates, detector, report_path)
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def print_results(stats, video_files, freeze_candidates, detector, report_path):
    """Print comprehensive analysis results"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE FREEZE ANALYSIS RESULTS")
    print("=" * 70)
    
    # Get total freeze statistics
    total_freeze_stats = detector.count_total_freezes()
    
    # Show top longest sequences per camera (length >= 2)
    camera_sequences = detector.get_top_longest_sequences(top_n=15, min_length=2)
    
    if camera_sequences:
        print(f"\nTop {min(15, len(camera_sequences))} Longest Freeze Sequences by Camera:")
        print("-" * 80)
        for i, seq in enumerate(camera_sequences[:15], 1):
            severity_icon = {
                'CRITICAL': '🔴',
                'HIGH': '🟠', 
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }.get(seq['severity'], '⚪')
            
            print(f"  {i:2d}. Camera {seq['camera']} | "
                  f"Frames {seq['start_frame']:3d}-{seq['end_frame']:3d} | "
                  f"Length: {seq['length']:2d} | "
                  f"Avg metric: {seq['avg_metric']:6.4f}")
    else:
        print(f"\nNo freeze sequences of length 2+ found.")
    
    # Camera analysis
    print(f"\nCamera Analysis:")
    for i, file_path in enumerate(video_files):
        camera_stat = detector.camera_statistics[i]
        freeze_count = camera_stat['freeze_count']
        freeze_pct = camera_stat['freeze_percentage']
        total_frames = camera_stat['total_frames']
        
        # Use same severity system as sequences
        if freeze_pct >= 10.0:
            severity = "CRITICAL 🔴"
        elif freeze_pct >= 5.0:
            severity = "HIGH 🟠"
        elif freeze_pct >= 2.0:
            severity = "MEDIUM 🟡"
        else:
            severity = "LOW 🟢"
        
        print(f"  Camera {i+1}: {freeze_count} freezes ({freeze_pct:.2f}%) of {total_frames} frames | {severity}")
    
    # Overall metrics
    setup_metrics = stats['setup_metrics']
    quality_score = setup_metrics['overall_quality_score']
    overall_freeze_rate = setup_metrics['overall_freeze_rate_percent']
    critical_sequences = setup_metrics['critical_sequences']
    
    # Use severity-based quality rating (inverted logic for quality)
    if overall_freeze_rate >= 10.0:
        quality_rating = "POOR 🔴"
    elif overall_freeze_rate >= 5.0:
        quality_rating = "BAD 🟠"
    elif overall_freeze_rate >= 2.0:
        quality_rating = "FAIR 🟡"
    else:
        quality_rating = "GOOD 🟢"
    
    # Calculate frame-based suspicious frames (any camera frozen in frame)
    frame_based_suspicious = sum(1 for frame in detector.frame_data if frame['frame_freeze'] == 1)
    frame_based_percentage = (frame_based_suspicious / detector.min_frames) * 100
    
    print(f"\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"📊 TOTAL FREEZE INSTANCES: {total_freeze_stats['total_freezes']} ({overall_freeze_rate:.2f}%)")
    print(f"📊 SUSPICIOUS FRAMES: {frame_based_suspicious} ({frame_based_percentage:.2f}%)")
    print(f"⚠️  CRITICAL PATTERNS: {critical_sequences} sequences (2+ frames)")
    print(f"🎯 QUALITY RATING: {quality_rating}")
    print(f"📁 IMAGES GENERATED: {len(freeze_candidates)} sets")
    print(f"📄 DATA SAVED: CSV, Excel, and JSON reports")
    print("=" * 70)


if __name__ == "__main__":
    main() 