import numpy as np
from pyglet.gl import *

class Light:
    def __init__(self, position: np.ndarray, color: np.ndarray, intensity: float = 1.0):
        self.position = position.astype('f4')
        self.color = color.astype('f4')
        self.intensity = float(intensity)

    def apply(self, light_id):
        """Applies the light settings to the given OpenGL light source."""
        glEnable(light_id)

        # Compute ambient, diffuse, and specular components
        ambient = (GLfloat * 4)(*(self.intensity * 0.2 * self.color).tolist() + [1.0])
        diffuse = (GLfloat * 4)(*(self.intensity * self.color).tolist() + [1.0])
        specular = (GLfloat * 4)(*(self.intensity * self.color).tolist() + [1.0])
        position = (GLfloat * 4)(*self.position.tolist(), 1.0)

        glLightfv(light_id, GL_AMBIENT, ambient)
        glLightfv(light_id, GL_DIFFUSE, diffuse)
        glLightfv(light_id, GL_SPECULAR, specular)
        glLightfv(light_id, GL_POSITION, position)
