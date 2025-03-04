import numpy as np

class Material:
    def __init__(self, 
                 albedo: np.ndarray = np.array([0.8, 0.8, 0.8]),
                 metallic: float = 0.0,
                 roughness: float = 0.5,
                 ambient: float = 0.2):
        self.albedo = albedo.astype('f4')
        self.metallic = float(metallic)
        self.roughness = float(roughness)
        self.ambient = float(ambient)
