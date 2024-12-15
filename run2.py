import torch
import torch.nn as nn
import numpy as np
import json

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
    
    # Denormalize latitude
    denorm_coords[:, 0] = (coords[:, 0] + 1) * (bounds['max_lat'] - bounds['min_lat']) / 2 + bounds['min_lat']
    
    # Denormalize longitude
    denorm_coords[:, 1] = (coords[:, 1] + 1) * (bounds['max_long'] - bounds['min_long']) / 2 + bounds['min_long']
    
    return denorm_coords

def main():
    # Load the model
    try:
        checkpoint = torch.load('dallas_crime_model.pt')
        model = CrimeDistributionModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        bounds = checkpoint['metadata']['dallas_bounds']
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set model to evaluation mode
    model.eval()

    # Generate random input data (100 samples)
    num_samples = 100
    input_data = torch.randn(num_samples, 10)  # 10 is input_dim

    # Generate predictions
    with torch.no_grad():
        try:
            predictions = model(input_data)
            predictions_np = predictions.numpy()
            
            # Denormalize the coordinates
            denormalized_coords = denormalize_coordinates(predictions_np, bounds)
            
            # Convert to list format matching the example
            coordinates_list = []
            for coords in denormalized_coords:
                coordinates_list.append([float(coords[0]), float(coords[1])])
            
            # Save as JSON file
            with open('data.json', 'w') as f:
                json.dump(coordinates_list, f)
            
            print(f"Generated {num_samples} predictions and saved to data.json")
            
            # Print first few predictions as example
            print("\nFirst 5 predicted locations (lat, long):")
            for i in range(min(5, len(coordinates_list))):
                print(f"Point {i+1}: {coordinates_list[i]}")
                
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()