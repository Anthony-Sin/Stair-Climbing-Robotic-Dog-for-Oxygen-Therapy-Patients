import time
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDConfig:
    """Configuration for PID controller"""
    kp: float = 0.8
    ki: float = 0.0
    kd: float = 0.4
    max_output: float = 0.6
    tolerance: float = 0.3
    antiwindup_gain: float = 0.0
    smoothing_alpha: float = 0.3


class PIDController:
    """
    A PID controller with deadband tolerance, anti-windup, and output smoothing.
    """
    
    def __init__(self, config: PIDConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset controller state"""
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_time = time.perf_counter()
        self.prev_output = 0.0
        self.prev_smoothed_output = 0.0
    
    def update(self, current_value: float, target_value: float) -> float:
        """
        Update the PID controller with current measurement
        
        Args:
            current_value: Current measured value
            target_value: Desired target value
            
        Returns:
            Control output (smoothed)
        """
        now = time.perf_counter()
        dt = max(1e-3, now - self.prev_time)
        
        error = current_value - target_value
        
        # Deadband: if within tolerance, output is zero and reset integral
        if abs(error) <= self.config.tolerance:
            output = 0.0
            self.integral_error = 0.0  # Reset integral when in deadband
        else:
            # Calculate derivative
            derivative = (error - self.prev_error) / dt
            
            # PID output before saturation
            pid_output = (self.config.kp * error + 
                         self.config.kd * derivative + 
                         self.config.ki * self.integral_error)
            
            # Saturate output
            max_output = max(0.0, self.config.max_output)
            output = float(np.clip(pid_output, -max_output, max_output))
            
            # Back-calculation anti-windup
            if self.config.antiwindup_gain > 0 and output != pid_output:
                # Adjust integral term to prevent windup
                self.integral_error += self.config.antiwindup_gain * (output - pid_output) * dt
            
            # Normal integral update
            self.integral_error += error * dt
            
            self.prev_error = error
        
        # Apply exponential moving average smoothing
        alpha = min(max(self.config.smoothing_alpha, 1e-3), 1.0)
        smoothed_output = self.prev_smoothed_output + alpha * (output - self.prev_smoothed_output)
        
        # Update state
        self.prev_time = now
        self.prev_output = output
        self.prev_smoothed_output = smoothed_output
        
        return smoothed_output
    
    def get_state(self) -> dict:
        """Get current controller state for debugging"""
        return {
            'prev_error': self.prev_error,
            'integral_error': self.integral_error,
            'prev_output': self.prev_output,
            'prev_smoothed_output': self.prev_smoothed_output
        }
