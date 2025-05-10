import numpy as np
from scipy.stats import qmc
import pickle
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

PARAM_BOUNDS = [
    (-10.0, 10.0),  # position_x
    (-10.0, 10.0),  # position_y
    (0.0, 5.0),     # position_z
    (-1.0, 1.0),    # direction_x
    (-1.0, 1.0),    # direction_y
    (-1.0, 1.0),    # direction_z
    (30.0, 120.0),  # horizontal_fov
    (20.0, 90.0),   # vertical_fov
    (0.05, 0.5)     # step_angle
]

class SurrogateOptimizer:
    def __init__(self, surrogate_model, simulator, dataset_path=None, device=None):
        """
        Surrogate-guided optimization for LiDAR sensor placement
        
        Args:
            surrogate_model: PyTorch model to predict sensor performance
            simulator: Interface to the LiDAR simulation system
            dataset_path: Path to load/save configuration dataset
            device: PyTorch device (cpu or cuda)
        """
        self.surrogate_model = surrogate_model
        self.simulator = simulator
        self.dataset_path = dataset_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs = []
        self.scores = []
        
        # Move model to device
        self.surrogate_model.to(self.device)
        
        # Load existing dataset if available
        if dataset_path and os.path.exists(dataset_path):
            self.load_dataset()
    
    def load_dataset(self):
        """Load existing dataset from file"""
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
            self.configs = data['configs']
            self.scores = data['scores']
    
    def save_dataset(self):
        """Save current dataset to file"""
        if self.dataset_path:
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            with open(self.dataset_path, 'wb') as f:
                pickle.dump({'configs': self.configs, 'scores': self.scores}, f)
    
    def initialize_dataset(self, num_samples=5, bounds=None):
        """Generate initial dataset with Latin Hypercube Sampling"""
        if not bounds:
            raise ValueError("Parameter bounds must be provided for initialization")
            
        # Create sampler for Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=len(bounds))
        samples = sampler.random(n=num_samples)
        
        # Extract lower and upper bounds for scaling
        l_bounds = [low for low, high in bounds]
        u_bounds = [high for low, high in bounds]
        
        # Scale all samples at once
        samples = qmc.scale(samples, l_bounds, u_bounds)
        
        # Evaluate each configuration
        for config in samples:
            score = self.simulator.evaluate_configuration(config)
            self.configs.append(config)
            self.scores.append(score)
            print(f"  Config evaluation: PCCS = {score:.4f}")
            
        # Save the dataset
        self.save_dataset()
    
    def train_surrogate(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train the surrogate model on current dataset using PyTorch"""
        # Convert data to PyTorch tensors
        X = torch.tensor(np.array(self.configs), dtype=torch.float32)
        y = torch.tensor(np.array(self.scores), dtype=torch.float32).view(-1, 1)
        
        # Split into train/validation
        dataset_size = len(X)
        indices = torch.randperm(dataset_size)
        split = int(np.floor(validation_split * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Create data loaders
        train_dataset = TensorDataset(X[train_indices], y[train_indices])
        val_dataset = TensorDataset(X[val_indices], y[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.surrogate_model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        # Training loop
        self.surrogate_model.train()
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.surrogate_model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # Validation phase
            self.surrogate_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.surrogate_model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                
            val_loss = val_loss / len(val_loader.dataset)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.surrogate_model.state_dict(), 'best_surrogate_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.surrogate_model.load_state_dict(torch.load('best_surrogate_model.pt'))
        return {'train_loss': train_loss, 'val_loss': val_loss}
    
    def predict_scores(self, configs):
        """Use surrogate model to predict scores for a batch of configurations"""
        self.surrogate_model.eval()
        with torch.no_grad():
            X = torch.tensor(configs, dtype=torch.float32).to(self.device)
            predictions = self.surrogate_model(X).cpu().numpy().flatten()
        return predictions
    
    def generate_candidates(self, candidates_per_iter):
        """
        Generate candidate configurations for surrogate-guided optimization.

        Args:
            candidates_per_iter (int): Number of candidates to generate.

        Returns:
            numpy.ndarray: A 2D array of shape (candidates_per_iter, n_dims)
                        with each dimension scaled according to PARAM_BOUNDS.
        """
        # Determine the number of dimensions from your parameter bounds.
        n_dims = len(PARAM_BOUNDS)
        
        # Initialize the QMC sampler if it hasn't been set yet.
        if not hasattr(self, 'qmc_sampler'):
            # Using LatinHypercube as an example; you can choose a different sampler if needed.
            self.qmc_sampler = qmc.LatinHypercube(d=n_dims)
        
        # Generate the candidate samples. The output will have shape (candidates_per_iter, n_dims).
        explore_samples = self.qmc_sampler.random(candidates_per_iter)
        
        # Scale each dimension of the samples to the corresponding bounds.
        for i in range(n_dims):
            low, high = PARAM_BOUNDS[i]
            # Reshape the 1D column to 2D, scale it, and then flatten back to 1D.
            scaled = qmc.scale(explore_samples[:, i].reshape(-1, 1), low, high)
            explore_samples[:, i] = scaled.flatten()
        
        return explore_samples

    
    def optimize(self, iterations=10, candidates_per_iter=1000, evals_per_iter=5):
        """Run the optimization loop"""
        best_config = None
        best_score = -float('inf')
        
        for iteration in range(iterations):
            print(f"Iteration {iteration+1}/{iterations}")
            
            # Generate candidate configurations
            candidates = self.generate_candidates(candidates_per_iter)
            
            # Predict scores using surrogate model
            predicted_scores = self.predict_scores(candidates)
            
            # Select top candidates for evaluation
            top_indices = np.argsort(predicted_scores)[-evals_per_iter:]
            top_candidates = candidates[top_indices]
            
            # Evaluate selected candidates with full simulation
            for config in top_candidates:
                score = self.simulator.evaluate_configuration(config)
                self.configs.append(config)
                self.scores.append(score)
                
                print(f"  New configuration evaluated: PCCS = {score:.4f}")
                
                # Update best configuration if needed
                if score > best_score:
                    best_score = score
                    best_config = config
            
            # Save dataset after each iteration
            self.save_dataset()
                
            # Re-train surrogate model with expanded dataset
            print("  Re-training surrogate model...")
            self.train_surrogate()
            
            # Print current best
            print(f"  Current best configuration: PCCS = {best_score:.4f}")
        
        return best_config, best_score