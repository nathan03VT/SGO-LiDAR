import numpy as np
from pyglet.gl import *

class Light:
    def __init__(self, position: np.ndarray, color: np.ndarray, intensity: float = 1.0):
        self.position = position.astype('f4')
        self.color = color.astype('f4')
        self.intensity = float(intensity)
