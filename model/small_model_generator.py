
import torch
import torch.nn as nn
import numpy as np
import os

class DallasCrimePredictor(nn.Module):
    def __init__(self):
        super(DallasCrimePredictor, self).__init__()
        
        # Reduced dimensions to achieve ~80MB model size
        self.input_dim = 32  # Reduced from 50
        self.hidden_dim1 = 1024  # Reduced from 2048
        self.hidden_dim2 = 2048  # Reduced from 4096
        self.hidden_dim3 = 1536  # Reduced from 3072
        self.output_locations = 500  # Reduced from 1000
        self.embedding_dim = 128  # Reduced from 256
        
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
        
        # Reduced number of auxiliary networks and their sizes
        self.auxiliary_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim3, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 1024)
            ) for _ in range(2)  # Reduced from 4 to 2 auxiliary networks
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

def initialize_model():
    """Initialize the model and print its size"""
    model = DallasCrimePredictor()
    
    # Initialize weights using Xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Calculate and print model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f'Model size: {size_mb:.2f} MB')
    
    return model

def save_model(model, save_path='dallas_crime_model.pt'):
    """Save the model with metadata"""
    # Define Dallas geographical bounds
    dallas_bounds = {
        'min_lat': 32.6185,
        'max_lat': 33.0198,
        'min_long': -96.9990,
        'max_long': -96.5738
    }
    
    # Save model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': {
            'dallas_bounds': dallas_bounds,
            'model_info': {
                'input_dim': model.input_dim,
                'output_locations': model.output_locations,
                'creation_date': '2024-12-15'
            }
        }
    }, save_path)
    print(f'Model saved to {save_path}')

if __name__ == "__main__":
    # Initialize model
    model = initialize_model()
    
    # Example input
    batch_size = 4
    x = torch.randn(batch_size, model.input_dim)
    day = torch.tensor([1, 100, 200, 365])  # Example days of year
    
    # Get predictions
    with torch.no_grad():
        predictions = model(x, day)
        
    # Print sample output
    print("\nSample predictions:")
    print(f"Locations shape: {predictions['locations'].shape}")
    print(f"Counts shape: {predictions['counts'].shape}")
    
    # Save model
    save_model(model)