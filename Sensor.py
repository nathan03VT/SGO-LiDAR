import numpy as np
from typing import Optional, Tuple

class Sensor:
    def __init__(
        self,
        position: np.ndarray,
        target_direction: Optional[np.ndarray] = None,
        horizontal_fov: float = 90.0,
        vertical_fov: float = 60.0,
        step_angle: float = 0.1,
        min_distance: float = 0.1,
        max_distance: float = 130.0
    ):
        """
        Initialize a sensor with position, direction and parameters
        
        Args:
            position: (3,) array representing sensor position [x, y, z]
            target_direction: Optional (3,) array representing direction vector
            horizontal_fov: Horizontal field of view in degrees (default: 90.0)
            vertical_fov: Vertical field of view in degrees (default: 60.0)
            step_angle: Angular resolution in degrees (default: 0.1)
            min_distance: Minimum sensing distance in meters (default: 0.1)
            max_distance: Maximum sensing distance in meters (default: 130.0)
        """
        # Set position
        self.position = np.asarray(position, dtype=np.float32)
        
        # Set target direction (default to pointing downward if none provided)
        if target_direction is None:
            self.target_direction = np.array([0, 0, -1], dtype=np.float32)
        else:
            self.target_direction = self._normalize_direction(target_direction)
            
        # Set LiDAR parameters
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.step_angle = step_angle
        self.min_distance = min_distance
        self.max_distance = max_distance
        
    def _normalize_direction(self, direction: np.ndarray) -> np.ndarray:
        """Normalize the direction vector"""
        direction = np.asarray(direction, dtype=np.float32)
        return direction / np.linalg.norm(direction)
    
    def set_position(self, position: np.ndarray) -> None:
        """Update sensor position"""
        self.position = np.asarray(position, dtype=np.float32)
        
    def set_target_direction(self, target_direction: np.ndarray) -> None:
        """Update sensor target direction"""
        self.target_direction = self._normalize_direction(target_direction)
        
    def update_params(self, **kwargs) -> None:
        """
        Update specific sensor parameters
        
        Example:
            sensor.update_params(horizontal_fov=120.0, step_angle=0.2)
        """
        valid_params = {
            'horizontal_fov', 'vertical_fov', 'step_angle',
            'min_distance', 'max_distance'
        }
        
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
                
    def get_sensor_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current sensor position and direction
        
        Returns:
            Tuple of (position, target_direction)
        """
        return self.position, self.target_direction
    
    def get_params(self) -> dict:
        """
        Get the current sensor parameters
        
        Returns:
            Dictionary containing all sensor parameters
        """
        return {
            'horizontal_fov': self.horizontal_fov,
            'vertical_fov': self.vertical_fov,
            'step_angle': self.step_angle,
            'min_distance': self.min_distance,
            'max_distance': self.max_distance
        }