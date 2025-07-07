import numpy as np
from typing import List, Tuple, Dict, Any
from numpy.typing import NDArray


class FreezeDetectorJerk:
    """
    Freeze detector using 3rd derivative (jerk) - the best performing metric
    """
    
    def __init__(self, frame_differences: List[List[float]], threshold_percent: float = 5.0):
        """
        Initialize jerk-based freeze detector
        
        Args:
            frame_differences: List of frame differences for each camera
            threshold_percent: Percentage of most suspicious frames to report
        """
        self.frame_differences = frame_differences
        self.threshold_percent = threshold_percent
        self.num_cameras = len(frame_differences)
        
        # Compute derivatives
        self.first_derivatives = []
        self.second_derivatives = []
        self.third_derivatives = []
        
        for diff_list in frame_differences:
            first_deriv = self._compute_first_derivative(diff_list)
            second_deriv = self._compute_second_derivative(diff_list)
            third_deriv = self._compute_third_derivative(second_deriv)
            self.first_derivatives.append(first_deriv)
            self.second_derivatives.append(second_deriv)
            self.third_derivatives.append(third_deriv)
        
        # Find minimum length for alignment
        self.min_length = min(len(deriv) for deriv in self.third_derivatives)
        
        # Compute jerk metrics
        self.jerk_metrics = self._compute_jerk_metrics()
        
    def _compute_first_derivative(self, differences: List[float]) -> List[float]:
        """Compute 1st derivative (velocity) - differences are already 1st derivative"""
        # Frame differences are already the 1st derivative (velocity)
        return differences
    
    def _compute_second_derivative(self, differences: List[float]) -> List[float]:
        """Compute 2nd derivative (acceleration) - derivative of velocity"""
        second_derivative = []
        for i in range(1, len(differences)):
            accel = differences[i] - differences[i-1]
            second_derivative.append(accel)
        return second_derivative
    
    def _compute_third_derivative(self, second_derivative: List[float]) -> List[float]:
        """Compute 3rd derivative (jerk)"""
        third_derivative = []
        for i in range(1, len(second_derivative)):
            jerk = second_derivative[i] - second_derivative[i-1]
            third_derivative.append(jerk)
        return third_derivative
    
    def _compute_jerk_metrics(self) -> List[Dict[str, Any]]:
        """Compute jerk metrics for each frame"""
        metrics = []
        
        for i in range(self.min_length):
            # Get velocity values for all cameras at this frame
            # Now first_derivatives has same length as frame_differences, so adjust indexing
            velocities = [self.first_derivatives[cam][i+2] for cam in range(self.num_cameras)]  # i+2 to align with jerk indices
            
            # Get jerk values for all cameras at this frame
            jerks = [self.third_derivatives[cam][i] for cam in range(self.num_cameras)]
            
            # Use same indexing as test script
            diff_index_3rd = i + 2      # For 3rd derivative (1 frame earlier) - this is what test script uses for known_freezes check
            frame_index = diff_index_3rd # Use diff_index_3rd as frame_index to match test results
            
            # Compute metric_3rd_max (best performing metric)
            metric_3rd_max = max(jerks)
            
            # Additional metrics for analysis
            min_jerk = min(jerks)
            abs_jerks = [abs(j) for j in jerks]
            max_abs_jerk = max(abs_jerks)
            jerk_range = max(jerks) - min(jerks)
            
            # Determine most suspicious camera
            max_jerk_cam = jerks.index(max(jerks))
            
            metrics.append({
                'frame_index': frame_index,
                'diff_index_3rd': diff_index_3rd,
                'velocities': velocities,
                'jerks': jerks,
                'metric_3rd_max': metric_3rd_max,
                'min_jerk': min_jerk,
                'max_abs_jerk': max_abs_jerk,
                'jerk_range': jerk_range,
                'most_suspicious_camera': max_jerk_cam,
                'velocity_cam_1': velocities[0],
                'velocity_cam_2': velocities[1],
                'velocity_cam_3': velocities[2],
                'jerk_cam_1': jerks[0],
                'jerk_cam_2': jerks[1],
                'jerk_cam_3': jerks[2]
            })
        
        return metrics
    
    def detect_freezes(self) -> List[Dict[str, Any]]:
        """
        Detect freeze candidates using jerk metric
        
        Returns:
            List of freeze candidates sorted by suspicion level
        """
        # Sort by metric_3rd_max (descending - higher jerk = more suspicious)
        sorted_metrics = sorted(self.jerk_metrics, key=lambda x: x['metric_3rd_max'], reverse=True)
        
        # Take top percentage as freeze candidates
        num_candidates = max(1, int(len(sorted_metrics) * self.threshold_percent / 100))
        freeze_candidates = sorted_metrics[:num_candidates]
        
        # Add ranking and suspicion score
        for i, candidate in enumerate(freeze_candidates):
            candidate['rank'] = i + 1
            candidate['suspicion_score'] = candidate['metric_3rd_max']
            candidate['detection_method'] = 'jerk_3rd_derivative'
        
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
            # Get frame differences for this camera
            differences = self.frame_differences[cam_idx]
            jerks = self.third_derivatives[cam_idx]
            
            # Calculate statistics
            avg_diff = np.mean(differences)
            median_diff = np.median(differences)
            std_diff = np.std(differences)
            
            avg_jerk = np.mean([abs(j) for j in jerks])
            max_jerk = max([abs(j) for j in jerks])
            
            # Count how many times this camera was most suspicious
            suspicious_count = sum(1 for metric in self.jerk_metrics 
                                 if metric['most_suspicious_camera'] == cam_idx)
            
            camera_stats.append({
                'camera_index': cam_idx,
                'avg_difference': float(avg_diff),
                'median_difference': float(median_diff),
                'std_difference': float(std_diff),
                'avg_abs_jerk': float(avg_jerk),
                'max_abs_jerk': float(max_jerk),
                'suspicious_count': suspicious_count,
                'freeze_count': suspicious_count,  # For compatibility
                'freeze_percentage': (suspicious_count / len(self.jerk_metrics)) * 100
            })
        
        # Overall metrics
        all_jerks = []
        for jerks in self.third_derivatives:
            all_jerks.extend([abs(j) for j in jerks])
        
        avg_jerk = np.mean(all_jerks)
        max_jerk = max(all_jerks)
        
        # Find most problematic camera
        most_problematic = max(range(self.num_cameras), 
                             key=lambda i: camera_stats[i]['suspicious_count'])
        
        # Setup quality metrics
        jerk_stability = 1.0 / (1.0 + np.std(all_jerks))
        freeze_rate = len(self.detect_freezes()) / len(self.jerk_metrics)
        
        setup_stability_score = min(100, jerk_stability * 100)
        overall_quality_score = max(0, 100 - (freeze_rate * 100 * 10))  # Penalize high freeze rate
        
        # Camera balance
        suspicious_counts = [stats['suspicious_count'] for stats in camera_stats]
        balance_coefficient = np.std(suspicious_counts) / (np.mean(suspicious_counts) + 1e-6)
        
        return {
            'detection_method': 'jerk_3rd_derivative',
            'total_frames': len(self.jerk_metrics),
            'total_differences': len(self.frame_differences[0]),
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
            'jerk_analysis': {
                'avg_absolute_jerk': float(avg_jerk),
                'max_absolute_jerk': float(max_jerk),
                'jerk_stability': float(jerk_stability)
            }
        } 