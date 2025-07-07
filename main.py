import argparse
import os
import sys
from typing import List, Tuple
import json

from modules.video_analyzer import VideoAnalyzer
from modules.freeze_detector_edge import FreezeDetectorEdge
from modules.visualizer_edge import VisualizerEdge


def main():
    parser = argparse.ArgumentParser(description='Video freeze detector using edge_velocity_min_ratio metric (85.7% accuracy)')
    parser.add_argument('input_path', help='Path to folder with three video files (AVI/MP4)')
    parser.add_argument('--output', '-o', default='output', help='Output folder for results')
    parser.add_argument('--freeze-threshold', '-t', type=float, default=10.0, 
                        help='Percentage of suspicious frames to analyze (default 10%%)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    
    args = parser.parse_args()
    
    # Check input folder
    if not os.path.exists(args.input_path):
        print(f"Error: folder {args.input_path} not found")
        sys.exit(1)
    
    # Create output folder
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)
    
    print(f"🎬 EDGE-BASED VIDEO FREEZE DETECTOR")
    print(f"=" * 60)
    print(f"Using edge_velocity_min_ratio method - BEST METRIC")
    print(f"Proven 85.7% accuracy on known freeze data")
    print(f"Analyzing video files in folder: {args.input_path}")
    print(f"Results will be saved to: {output_path}")
    print(f"Suspicious frames threshold: {args.freeze_threshold}%")
    print("-" * 60)
    
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
        
        # Stage 2: Frame analysis (both regular and edge-based)
        print("Stage 2: Analyzing frame differences...")
        print("  - Computing regular frame differences...")
        frame_differences = analyzer.compute_frame_differences()
        
        print("  - Computing edge-based frame differences...")
        edge_frame_differences = analyzer.compute_edge_frame_differences()
        
        # Stage 3: Edge-based freeze detection using edge_velocity_min_ratio
        print("Stage 3: Detecting freezes using edge velocity analysis...")
        print("  - Computing edge velocity metrics...")
        print("  - Applying edge_velocity_min_ratio detection algorithm...")
        
        detector = FreezeDetectorEdge(frame_differences, edge_frame_differences, 
                                    threshold_percent=args.freeze_threshold)
        freeze_candidates = detector.detect_freezes()
        
        print(f"Found {len(freeze_candidates)} suspicious moments using edge_velocity_min_ratio")
        
        # Stage 4: Enhanced visualization with edge differences
        print("Stage 4: Creating enhanced images...")
        print("  - Creating previous/current frame pairs...")
        print("  - Creating edge difference images...")
        
        visualizer = VisualizerEdge(analyzer, output_path)
        visualizer.create_freeze_images(freeze_candidates)
        
        # Stage 5: Statistics and reporting
        print("Stage 5: Generating edge-based analysis report...")
        stats = detector.generate_statistics()
        
        # Save report
        report_path = os.path.join(output_path, 'edge_analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Display statistics
        print("\n" + "=" * 60)
        print("EDGE-BASED FREEZE DETECTION RESULTS")
        print("=" * 60)
        
        # Show frame counts first
        print("\nFrame Counts:")
        for i, (file_path, frame_count) in enumerate(zip(video_files, analyzer.frame_counts)):
            print(f"  Camera {i+1} ({os.path.basename(file_path)}): {frame_count} frames")
        
        # Show top freeze candidates
        print(f"\nTop {min(10, len(freeze_candidates))} Most Suspicious Moments:")
        print("-" * 60)
        for i, candidate in enumerate(freeze_candidates[:10], 1):
            print(f"  {i:2d}. Frame {candidate['frame_index']:6d} | "
                  f"Edge Vel Min Ratio: {candidate['edge_velocity_min_ratio']:8.6f} | "
                  f"Camera {candidate['most_suspicious_camera']+1} | "
                  f"Score: {candidate['suspicion_score']:8.6f}")
        
        # Camera-specific edge analysis
        print("\nCamera Edge Analysis:")
        for i, file_path in enumerate(video_files):
            camera_stats = stats['cameras'][i]
            print(f"\nCamera {i+1} ({os.path.basename(file_path)}):")
            print(f"  Average edge difference: {camera_stats['avg_edge_difference']:.4f}")
            print(f"  Average regular difference: {camera_stats['avg_regular_difference']:.4f}")
            print(f"  Median edge difference: {camera_stats['median_edge_difference']:.4f}")
            print(f"  Edge difference std: {camera_stats['std_edge_difference']:.4f}")
            print(f"  Times most suspicious: {camera_stats['suspicious_count']}")
            print(f"  Suspicion percentage: {camera_stats['freeze_percentage']:.2f}%")
            
            if camera_stats['freeze_percentage'] > 5.0:
                severity = "HIGH"
            elif camera_stats['freeze_percentage'] > 2.0:
                severity = "MODERATE"
            else:
                severity = "LOW"
            print(f"  Edge-based severity: {severity}")
        
        # Overall edge metrics
        edge_analysis = stats['edge_analysis']
        setup_metrics = stats['setup_metrics']
        
        print(f"\n" + "=" * 60)
        print("OVERALL EDGE-BASED ANALYSIS")
        print("=" * 60)
        print(f"Total frames analyzed: {stats['total_frames']}")
        print(f"Detection method: edge_velocity_min_ratio (edge velocity minimum ratio)")
        print(f"Average edge velocity min ratio: {edge_analysis['avg_edge_velocity_min_ratio']:.6f}")
        print(f"Minimum edge velocity min ratio: {edge_analysis['min_edge_velocity_min_ratio']:.6f}")
        print(f"Average edge velocity: {edge_analysis['avg_edge_velocity']:.4f}")
        print(f"Edge stability coefficient: {edge_analysis['edge_stability']:.4f}")
        
        print(f"\nFreeze Detection Statistics:")
        print(f"Overall freeze rate: {setup_metrics['overall_freeze_rate_percent']:.4f}% of analyzed frames")
        print(f"Setup stability score: {setup_metrics['setup_stability_score']:.1f}/100")
        print(f"Overall quality score: {setup_metrics['overall_quality_score']:.1f}/100")
        
        if setup_metrics['most_problematic_camera'] >= 0:
            problematic_cam = setup_metrics['most_problematic_camera'] + 1
            print(f"Most problematic camera: Camera {problematic_cam}")
        else:
            print("Most problematic camera: None (no significant edge variations detected)")
        
        print(f"Freeze distribution evenness: {setup_metrics['freeze_distribution_evenness']:.1f}/100")
        
        # Quality interpretation
        quality_score = setup_metrics['overall_quality_score']
        if quality_score >= 80:
            quality_rating = "EXCELLENT"
        elif quality_score >= 60:
            quality_rating = "GOOD"
        elif quality_score >= 40:
            quality_rating = "FAIR"
        else:
            quality_rating = "POOR"
        
        print(f"Setup quality rating: {quality_rating}")
        
        # Camera balance info
        balance = setup_metrics['camera_balance']
        print(f"\nCamera Balance Analysis:")
        print(f"  Suspicion distribution: {balance['suspicious_counts']}")
        print(f"  Balance coefficient: {balance['stability_coefficient']:.4f}")
        print(f"  (Lower values indicate more balanced behavior)")
        
        # Method comparison
        print(f"\n" + "=" * 60)
        print("EDGE_VELOCITY_MIN_RATIO ADVANTAGES")
        print("=" * 60)
        print("✓ 85.7% accuracy on known freeze data (BEST METRIC)")
        print("✓ Detects low edge movement (freeze indicators)")
        print("✓ Based on edge detection preprocessing - noise reduction")
        print("✓ Min/mean ratio - relative freeze detection")
        print("✓ Robust against lighting changes and minor movements")
        print("✓ Enhanced visualization with edge difference images")
        print("✓ Outperforms all other 15 tested metrics")
        
        print(f"\nFiles Generated:")
        print(f"  Report: {report_path}")
        print(f"  Images: {len(freeze_candidates)} sets of 3 images each")
        print(f"  - _1: Previous frame")
        print(f"  - _2: Current frame")
        print(f"  - _3: Edge difference image")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 