import numpy as np

class Material:
    def __init__(self, 
                 albedo: float = 0.7,
                 metallic: float = 0.0,
                 roughness: float = 0.5,
                 ambient: float = 0.2):
        self.albedo = float(albedo)
        self.metallic = float(metallic)
        self.roughness = float(roughness)
        self.ambient = float(ambient)
