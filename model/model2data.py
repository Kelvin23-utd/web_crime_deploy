import torch
import torch.nn as nn
import json
import numpy as np
from datetime import datetime, timedelta
import os

class DallasCrimePredictor(nn.Module):
    def __init__(self):
        super(DallasCrimePredictor, self).__init__()
        
        # Model dimensions
        self.input_dim = 32
        self.hidden_dim1 = 1024
        self.hidden_dim2 = 2048
        self.hidden_dim3 = 1536
        self.output_locations = 500
        self.embedding_dim = 128
        
        # Embedding layer for processing temporal features
        self.temporal_embedding = nn.Embedding(366, self.embedding_dim)
        
        # Main network architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim + self.embedding_dim, self.hidden_dim1),
            nn.LayerNorm(self.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.LayerNorm(self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.LayerNorm(self.hidden_dim3),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Location prediction heads
        self.location_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim3, self.output_locations * 2),
            nn.Tanh()
        )
        
        # Crime count prediction head
        self.count_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim3, self.output_locations),
            nn.ReLU()
        )
        
        # Auxiliary networks
        self.auxiliary_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim3, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 1024)
            ) for _ in range(2)
        ])

    def forward(self, x, day_of_year):
        # Embed temporal features
        day_embedding = self.temporal_embedding(day_of_year)
        
        # Concatenate input features with temporal embedding
        combined_features = torch.cat([x, day_embedding], dim=1)
        
        # Extract features
        features = self.feature_extractor(combined_features)
        
        # Apply auxiliary networks
        aux_outputs = []
        for aux_net in self.auxiliary_networks:
            aux_outputs.append(aux_net(features))
        
        # Predict locations and crime counts
        locations = self.location_predictor(features)
        counts = self.count_predictor(features)
        
        # Reshape locations to (batch_size, num_locations, 2)
        locations = locations.view(-1, self.output_locations, 2)
        
        return {
            'locations': locations,
            'counts': counts,
            'aux_outputs': aux_outputs
        }
        
        

def generate_features(num_samples):
    """Generate random feature vectors for prediction"""
    return torch.randn(num_samples, 32)  # 32 is input_dim

def denormalize_coordinates(coords, bounds):
    """Convert coordinates from [-1,1] range back to actual lat/long"""
    denorm_coords = np.zeros_like(coords)
    
    # Denormalize latitude
    denorm_coords[:, 0] = (coords[:, 0] + 1) * (bounds['max_lat'] - bounds['min_lat']) / 2 + bounds['min_lat']
    
    # Denormalize longitude
    denorm_coords[:, 1] = (coords[:, 1] + 1) * (bounds['max_long'] - bounds['min_long']) / 2 + bounds['min_long']
    
    return denorm_coords

def generate_predictions(days=7):
    """Generate predictions for multiple days and save as JSON"""
    try:
        # Load the model
        model = DallasCrimePredictor()
        checkpoint = torch.load('model/dallas_crime_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        bounds = checkpoint['metadata']['dallas_bounds']
        model.eval()
        
        print("Model loaded successfully")
        
        # Get current date
        current_date = datetime.now()
        
        # Generate predictions for each day
        all_predictions = []
        
        for day_offset in range(days):
            # Calculate date and day of year
            prediction_date = current_date + timedelta(days=day_offset)
            day_of_year = prediction_date.timetuple().tm_yday
            
            # Generate input features
            features = generate_features(1)  # Generate for one day
            day_tensor = torch.tensor([day_of_year])
            
            print(f"Generating predictions for {prediction_date.strftime('%Y-%m-%d')}")
            
            # Get predictions
            with torch.no_grad():
                predictions = model(features, day_tensor)
                
                # Get numpy arrays
                locations = predictions['locations'][0].numpy()  # First batch only
                counts = predictions['counts'][0].numpy()
                
                # Denormalize coordinates
                locations = denormalize_coordinates(locations, bounds)
                
                # Convert predictions to list of dictionaries
                day_predictions = []
                for loc_idx in range(len(locations)):
                    if counts[loc_idx] > 0:  # Only include non-zero predictions
                        prediction = {
                            'date': prediction_date.strftime('%Y-%m-%d'),
                            'latitude': float(locations[loc_idx][0]),
                            'longitude': float(locations[loc_idx][1]),
                            'predicted_crimes': int(counts[loc_idx]),
                        }
                        day_predictions.append(prediction)
                
                all_predictions.extend(day_predictions)
        
        # Save predictions to JSON
        output_data = {
            'generated_at': current_date.strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': all_predictions,
            'metadata': {
                'num_days': days,
                'bounds': bounds
            }
        }
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Save the JSON file
        output_path = os.path.join('output', 'crime_predictions.json')
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"\nGenerated predictions for {days} days and saved to {output_path}")
        print(f"Total predictions: {len(all_predictions)}")
        
    except Exception as e:
        print(f"Error generating predictions: {e}")

if __name__ == "__main__":
    # Generate predictions for the next 7 days
    generate_predictions(days=7)