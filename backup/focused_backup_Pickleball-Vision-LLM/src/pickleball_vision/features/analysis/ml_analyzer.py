"""Machine learning analyzer for pickleball game analysis."""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
from ..utils.logger import setup_logger

class ShotPredictor(nn.Module):
    """Neural network for shot prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize the shot predictor.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Size of output layer
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.network(x)

class MLAnalyzer:
    """Machine learning analyzer for pickleball game analysis."""
    
    def __init__(self):
        """Initialize the ML analyzer."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger()
        self.scaler = StandardScaler()
        self.shot_predictor = None
        
    def train_shot_predictor(self, shots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train shot predictor model.
        
        Args:
            shots: List of shot data
            
        Returns:
            Training metrics
        """
        # Prepare training data
        features = np.array([
            [
                s['placement_x'],
                s['placement_y'],
                s['speed'],
                s['spin'],
                s['player_position_x'],
                s['player_position_y'],
                s['opponent_position_x']
            ]
            for s in shots
        ])
        
        labels = np.array([s['success'] for s in shots])
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        # Convert to tensors
        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels, dtype=torch.float32).to(self.device)
        
        # Initialize model
        self.shot_predictor = ShotPredictor(
            input_size=features.shape[1],
            hidden_size=64,
            output_size=1
        ).to(self.device)
        
        # Train model
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.shot_predictor.parameters())
        
        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.shot_predictor(X)
            loss = criterion(output, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        
        return {
            'final_loss': losses[-1],
            'loss_history': losses
        }
    
    def predict_shot_success(self, shot: Dict[str, Any]) -> float:
        """Predict shot success probability.
        
        Args:
            shot: Shot data
            
        Returns:
            Success probability
        """
        if self.shot_predictor is None:
            raise ValueError("Shot predictor not trained")
        
        # Prepare features
        features = np.array([[
            shot['placement_x'],
            shot['placement_y'],
            shot['speed'],
            shot['spin'],
            shot['player_position_x'],
            shot['player_position_y'],
            shot['opponent_position_x']
        ]])
        
        # Scale features
        features = self.scaler.transform(features)
        
        # Convert to tensor
        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.shot_predictor(X)
        
        return float(output[0][0])
    
    def analyze_player_style(self, shots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze player's playing style.
        
        Args:
            shots: List of shot data
            
        Returns:
            Style analysis results
        """
        # Calculate shot type distribution
        shot_types = [s['shot_type'] for s in shots]
        type_counts = {
            shot_type: shot_types.count(shot_type)
            for shot_type in set(shot_types)
        }
        
        # Calculate average metrics
        avg_speed = np.mean([s['speed'] for s in shots])
        avg_spin = np.mean([s['spin'] for s in shots])
        success_rate = np.mean([s['success'] for s in shots])
        
        return {
            'shot_distribution': type_counts,
            'avg_speed': float(avg_speed),
            'avg_spin': float(avg_spin),
            'success_rate': float(success_rate)
        } 