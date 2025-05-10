import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SurrogateModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128, 64, 32]):
        """
        Create the PyTorch neural network surrogate model
        
        Args:
            input_dim: Number of input features (depends on sensor config representation)
            hidden_layers: List defining the size of hidden layers
        """
        super(SurrogateModel, self).__init__()
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.Dropout(0.2))
            prev_dim = units
        
        # Output layer (PCCS prediction)
        layers.append(nn.Linear(prev_dim, 1))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# More complex model with attention for sensor interaction
class AttentionSurrogateModel(nn.Module):
    def __init__(self, num_sensors, features_per_sensor):
        """
        Create surrogate model that handles sensor interactions with self-attention
        
        Args:
            num_sensors: Number of sensors to configure
            features_per_sensor: Number of features describing each sensor
        """
        super(AttentionSurrogateModel, self).__init__()
        
        self.num_sensors = num_sensors
        self.features_per_sensor = features_per_sensor
        
        # Feature extraction for each sensor
        self.feature_extraction = nn.Linear(features_per_sensor, 32)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(32)
        
        # Output layers
        self.fc1 = nn.Linear(32 * num_sensors, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        # Reshape input to separate sensors
        x = x.view(-1, self.num_sensors, self.features_per_sensor)
        
        # Extract features for each sensor
        sensor_features = F.relu(self.feature_extraction(x))
        
        # Apply self-attention
        attn_output, _ = self.attention(sensor_features, sensor_features, sensor_features)
        
        # Add residual connection and normalize
        sensor_features = self.layer_norm(sensor_features + attn_output)
        
        # Flatten and pass through output layers
        x = sensor_features.reshape(-1, 32 * self.num_sensors)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.output(x)
        
        return x

def create_surrogate_model(input_dim, hidden_layers=[128, 64, 32]):
    """Factory function to create and initialize the model"""
    model = SurrogateModel(input_dim, hidden_layers)
    return model

def create_attention_surrogate_model(num_sensors, features_per_sensor):
    """Factory function to create attention-based model"""
    model = AttentionSurrogateModel(num_sensors, features_per_sensor)
    return model