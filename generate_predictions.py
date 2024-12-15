import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path

class CrimeDistributionModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        super(CrimeDistributionModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

def denormalize_coordinates(coords, bounds):
    """Convert coordinates from [-1,1] range back to actual lat/long"""
    denorm_coords = np.zeros_like(coords)
    denorm_coords[:, 0] = (coords[:, 0] + 1) * (bounds['max_lat'] - bounds['min_lat']) / 2 + bounds['min_lat']
    denorm_coords[:, 1] = (coords[:, 1] + 1) * (bounds['max_long'] - bounds['min_long']) / 2 + bounds['min_long']
    return denorm_coords

def main():
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    # Load the model
    try:
        checkpoint = torch.load('dallas_crime_model.pt', map_location=torch.device('cpu'))
        model = CrimeDistributionModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        bounds = checkpoint['metadata']['dallas_bounds']
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")

    # Set model to evaluation mode
    model.eval()

    # Generate predictions
    try:
        num_samples = 100
        input_data = torch.randn(num_samples, 10)
        
        with torch.no_grad():
            predictions = model(input_data)
            predictions_np = predictions.numpy()
            denormalized_coords = denormalize_coordinates(predictions_np, bounds)
            
            # Convert to list format
            coordinates_list = [[float(lat), float(lon)] for lat, lon in denormalized_coords]
            
            # Save as JSON
            with open('data.json', 'w') as f:
                json.dump(coordinates_list, f, indent=2)
                
    except Exception as e:
        raise Exception(f"Failed to generate predictions: {e}")

if __name__ == "__main__":
    main()