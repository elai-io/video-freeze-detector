import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
from numpy.typing import NDArray


class FreezeDetectorEdge:
    """
    Freeze detector using edge_velocity_min_ratio - the best performing metric (85.7% accuracy)
    """
    
    def __init__(self, frame_differences: List[List[float]], 
                 edge_frame_differences: List[List[float]], 
                 threshold_percent: float = 5.0):
        """
        Initialize edge-based freeze detector
        
        Args:
            frame_differences: List of regular frame differences for each camera
            edge_frame_differences: List of edge-based frame differences for each camera
            threshold_percent: Percentage of most suspicious frames to report
        """
        self.frame_differences = frame_differences
        self.edge_frame_differences = edge_frame_differences
        self.threshold_percent = threshold_percent
        self.num_cameras = len(frame_differences)
        
        # Compute edge velocity metrics
        self.edge_velocity_metrics = self._compute_edge_velocity_metrics()
        
    def _compute_edge_velocity_metrics(self) -> List[Dict[str, Any]]:
        """Compute edge velocity metrics for each frame"""
        metrics = []
        
        # Find minimum length across all cameras
        min_length = min(len(diffs) for diffs in self.edge_frame_differences)
        
        for i in range(min_length):
            # Get edge velocities for all cameras at this frame
            edge_velocities = [self.edge_frame_differences[cam][i] for cam in range(self.num_cameras)]
            
            # Calculate edge_velocity_min_ratio (the winning metric)
            min_edge_velocity = min(edge_velocities)
            mean_edge_velocity = np.mean(edge_velocities)
            
            # Avoid division by zero
            if mean_edge_velocity > 0:
                edge_velocity_min_ratio = min_edge_velocity / mean_edge_velocity
            else:
                edge_velocity_min_ratio = 0.0
            
            # Determine most suspicious camera (lowest edge velocity = most likely frozen)
            min_edge_velocity_cam = edge_velocities.index(min_edge_velocity)
            
            metrics.append({
                'frame_index': i + 1,  # Frame differences[0] corresponds to frame 1
                'edge_velocities': edge_velocities,
                'edge_velocity_min_ratio': edge_velocity_min_ratio,  # Main metric
                'most_suspicious_camera': min_edge_velocity_cam
            })
        
        return metrics
    
    def detect_freezes(self) -> List[Dict[str, Any]]:
        """
        Detect freeze candidates using edge_velocity_min_ratio
        
        Returns:
            List of freeze candidates sorted by suspicion level
        """
        # Sort by edge_velocity_min_ratio (ascending - lower ratio = more suspicious)
        sorted_metrics = sorted(self.edge_velocity_metrics, 
                              key=lambda x: x['edge_velocity_min_ratio'], 
                              reverse=False)
        
        # Take top percentage as freeze candidates
        num_candidates = max(1, int(len(sorted_metrics) * self.threshold_percent / 100))
        freeze_candidates = sorted_metrics[:num_candidates]
        
        # Add ranking and suspicion score
        for i, candidate in enumerate(freeze_candidates):
            candidate['rank'] = i + 1
            candidate['suspicion_score'] = 1.0 - candidate['edge_velocity_min_ratio']  # Lower ratio = higher suspicion
            candidate['detection_method'] = 'edge_velocity_min_ratio'
        
        return freeze_candidates
    
    def generate_statistics(self) -> Dict[str, Any]:
        """
        Generate detailed statistics about the analysis
        
        Returns:
            Dictionary with analysis statistics
        """
        # Camera-specific stats
        camera_stats = []
        for cam_idx in range(self.num_cameras):
            # Get edge differences for this camera
            edge_differences = self.edge_frame_differences[cam_idx]
            regular_differences = self.frame_differences[cam_idx]
            
            # Calculate statistics
            avg_edge_diff = np.mean(edge_differences)
            median_edge_diff = np.median(edge_differences)
            std_edge_diff = np.std(edge_differences)
            
            avg_regular_diff = np.mean(regular_differences)
            
            # Count how many times this camera was most suspicious
            suspicious_count = sum(1 for metric in self.edge_velocity_metrics 
                                 if metric['most_suspicious_camera'] == cam_idx)
            
            camera_stats.append({
                'camera_index': cam_idx,
                'avg_edge_difference': float(avg_edge_diff),
                'median_edge_difference': float(median_edge_diff),
                'std_edge_difference': float(std_edge_diff),
                'avg_regular_difference': float(avg_regular_diff),
                'suspicious_count': suspicious_count,
                'freeze_count': suspicious_count,  # For compatibility
                'freeze_percentage': (suspicious_count / len(self.edge_velocity_metrics)) * 100
            })
        
        # Overall metrics
        all_edge_ratios = [metric['edge_velocity_min_ratio'] for metric in self.edge_velocity_metrics]
        all_edge_velocities = []
        for edge_diffs in self.edge_frame_differences:
            all_edge_velocities.extend(edge_diffs)
        
        avg_edge_ratio = np.mean(all_edge_ratios)
        min_edge_ratio = min(all_edge_ratios)
        avg_edge_velocity = np.mean(all_edge_velocities)
        
        # Find most problematic camera
        most_problematic = max(range(self.num_cameras), 
                             key=lambda i: camera_stats[i]['suspicious_count'])
        
        # Setup quality metrics
        edge_stability = 1.0 / (1.0 + np.std(all_edge_ratios))
        freeze_rate = len(self.detect_freezes()) / len(self.edge_velocity_metrics)
        
        setup_stability_score = min(100, edge_stability * 100)
        overall_quality_score = max(0, 100 - (freeze_rate * 100 * 10))  # Penalize high freeze rate
        
        # Camera balance
        suspicious_counts = [stats['suspicious_count'] for stats in camera_stats]
        balance_coefficient = np.std(suspicious_counts) / (np.mean(suspicious_counts) + 1e-6)
        
        return {
            'detection_method': 'edge_velocity_min_ratio',
            'total_frames': len(self.edge_velocity_metrics),
            'total_differences': len(self.edge_frame_differences[0]),
            'cameras': camera_stats,
            'setup_metrics': {
                'overall_freeze_rate_percent': freeze_rate * 100,
                'setup_stability_score': float(setup_stability_score),
                'overall_quality_score': float(overall_quality_score),
                'most_problematic_camera': most_problematic,
                'freeze_distribution_evenness': max(0, 100 - balance_coefficient * 50),
                'camera_balance': {
                    'stability_coefficient': float(balance_coefficient),
                    'suspicious_counts': suspicious_counts
                }
            },
            'edge_analysis': {
                'avg_edge_velocity_min_ratio': float(avg_edge_ratio),
                'min_edge_velocity_min_ratio': float(min_edge_ratio),
                'avg_edge_velocity': float(avg_edge_velocity),
                'edge_stability': float(edge_stability)
            }
        } 