import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from ..utils.logger import setup_logger
from .shot_analyzer import ShotAnalyzer, ShotMetrics

logger = setup_logger(__name__)

@dataclass
class PlayerStats:
    total_shots: int
    shot_distribution: Dict[str, int]
    average_speed: float
    average_placement: Tuple[float, float]
    movement_distance: float
    max_speed: float
    court_coverage: float
    shot_effectiveness: float
    rally_win_percentage: float
    error_rate: float

class GameAnalyzer:
    def __init__(self):
        self.shot_analyzer = ShotAnalyzer()
        self.court_dimensions = {
            'length': 44,  # feet
            'width': 20,   # feet
            'net_height': 3  # feet
        }
    
    def analyze_game(self, 
                    shot_data: pd.DataFrame,
                    player_data: pd.DataFrame,
                    rally_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze complete game data."""
        try:
            # Analyze each player
            player_stats = {}
            for player in player_data['player_id'].unique():
                player_stats[player] = self._analyze_player(
                    player,
                    shot_data[shot_data['player_id'] == player],
                    player_data[player_data['player_id'] == player],
                    rally_data
                )
            
            # Analyze team dynamics
            team_stats = self._analyze_team_dynamics(player_data, rally_data)
            
            # Analyze game patterns
            game_patterns = self._analyze_game_patterns(shot_data, rally_data)
            
            return {
                'player_stats': player_stats,
                'team_stats': team_stats,
                'game_patterns': game_patterns
            }
            
        except Exception as e:
            logger.error(f"Error analyzing game: {str(e)}")
            raise
    
    def _analyze_player(self,
                       player_id: str,
                       shots: pd.DataFrame,
                       movements: pd.DataFrame,
                       rallies: pd.DataFrame) -> PlayerStats:
        """Analyze individual player performance."""
        # Calculate shot statistics
        shot_distribution = shots['shot_type'].value_counts().to_dict()
        average_speed = shots['speed'].mean()
        average_placement = (
            shots['placement_x'].mean(),
            shots['placement_y'].mean()
        )
        
        # Calculate movement statistics
        movement_distance = self._calculate_total_distance(movements)
        max_speed = movements['speed'].max()
        court_coverage = self._calculate_court_coverage(movements)
        
        # Calculate effectiveness
        shot_effectiveness = shots['effectiveness_score'].mean()
        rally_win_percentage = self._calculate_rally_win_percentage(player_id, rallies)
        error_rate = self._calculate_error_rate(shots)
        
        return PlayerStats(
            total_shots=len(shots),
            shot_distribution=shot_distribution,
            average_speed=average_speed,
            average_placement=average_placement,
            movement_distance=movement_distance,
            max_speed=max_speed,
            court_coverage=court_coverage,
            shot_effectiveness=shot_effectiveness,
            rally_win_percentage=rally_win_percentage,
            error_rate=error_rate
        )
    
    def _analyze_team_dynamics(self,
                             player_data: pd.DataFrame,
                             rally_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze team dynamics and coordination."""
        # Calculate team coverage
        team_coverage = self._calculate_team_coverage(player_data)
        
        # Analyze player positioning
        positioning_analysis = self._analyze_player_positioning(player_data)
        
        # Calculate team effectiveness
        team_effectiveness = self._calculate_team_effectiveness(rally_data)
        
        return {
            'team_coverage': team_coverage,
            'positioning_analysis': positioning_analysis,
            'team_effectiveness': team_effectiveness
        }
    
    def _analyze_game_patterns(self,
                             shot_data: pd.DataFrame,
                             rally_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze game patterns and strategies."""
        # Analyze shot sequences
        shot_sequences = self._analyze_shot_sequences(shot_data)
        
        # Analyze rally patterns
        rally_patterns = self._analyze_rally_patterns(rally_data)
        
        # Analyze court usage
        court_usage = self._analyze_court_usage(shot_data)
        
        return {
            'shot_sequences': shot_sequences,
            'rally_patterns': rally_patterns,
            'court_usage': court_usage
        }
    
    def _calculate_total_distance(self, movements: pd.DataFrame) -> float:
        """Calculate total distance traveled by player."""
        return np.sum(movements['speed']) * movements['timestamp'].diff().mean()
    
    def _calculate_court_coverage(self, movements: pd.DataFrame) -> float:
        """Calculate percentage of court covered by player."""
        # Create court grid
        grid_size = 1  # feet
        x_bins = np.arange(0, self.court_dimensions['length'] + grid_size, grid_size)
        y_bins = np.arange(0, self.court_dimensions['width'] + grid_size, grid_size)
        
        # Calculate coverage
        H, _, _ = np.histogram2d(
            movements['x'],
            movements['y'],
            bins=[x_bins, y_bins]
        )
        
        covered_cells = np.sum(H > 0)
        total_cells = len(x_bins) * len(y_bins)
        
        return (covered_cells / total_cells) * 100
    
    def _calculate_rally_win_percentage(self,
                                      player_id: str,
                                      rallies: pd.DataFrame) -> float:
        """Calculate percentage of rallies won by player."""
        player_rallies = rallies[rallies['winner_id'] == player_id]
        return (len(player_rallies) / len(rallies)) * 100
    
    def _calculate_error_rate(self, shots: pd.DataFrame) -> float:
        """Calculate player error rate."""
        errors = shots[shots['effectiveness_score'] < 30]
        return (len(errors) / len(shots)) * 100
    
    def _calculate_team_coverage(self, player_data: pd.DataFrame) -> float:
        """Calculate team court coverage."""
        team_positions = player_data[['x', 'y']].values
        return self._calculate_court_coverage(pd.DataFrame({
            'x': team_positions[:, 0],
            'y': team_positions[:, 1]
        }))
    
    def _analyze_player_positioning(self, player_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze player positioning and court coverage."""
        # Calculate average positions
        avg_positions = player_data.groupby('player_id')[['x', 'y']].mean()
        
        # Calculate positioning effectiveness
        positioning_scores = {}
        for player_id, pos in avg_positions.iterrows():
            positioning_scores[player_id] = self._calculate_positioning_score(pos)
        
        return {
            'average_positions': avg_positions.to_dict(),
            'positioning_scores': positioning_scores
        }
    
    def _calculate_positioning_score(self, position: pd.Series) -> float:
        """Calculate effectiveness of player positioning."""
        # Define optimal zones
        optimal_zones = [
            (5, 2),   # Kitchen
            (15, 5),  # Back court
            (10, 8)   # Middle
        ]
        
        # Calculate distance to optimal zones
        distances = [
            np.sqrt((position['x'] - zx)**2 + (position['y'] - zy)**2)
            for zx, zy in optimal_zones
        ]
        
        return 100 - min(distances) * 10
    
    def _analyze_shot_sequences(self, shot_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in shot sequences."""
        # Group shots by rally
        rally_shots = shot_data.groupby('rally_id')
        
        # Analyze common sequences
        sequences = []
        for _, rally in rally_shots:
            sequences.append(rally['shot_type'].tolist())
        
        # Find common patterns
        pattern_analysis = self._find_common_patterns(sequences)
        
        return {
            'common_sequences': pattern_analysis,
            'sequence_effectiveness': self._calculate_sequence_effectiveness(sequences)
        }
    
    def _find_common_patterns(self, sequences: List[List[str]]) -> Dict[str, int]:
        """Find common shot patterns in sequences."""
        patterns = {}
        for seq in sequences:
            for i in range(len(seq) - 1):
                pattern = f"{seq[i]} -> {seq[i+1]}"
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        return patterns
    
    def _calculate_sequence_effectiveness(self, sequences: List[List[str]]) -> float:
        """Calculate effectiveness of shot sequences."""
        # This is a simplified calculation
        # In practice, you'd want a more sophisticated model
        return np.mean([len(seq) for seq in sequences])
    
    def _analyze_rally_patterns(self, rally_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in rallies."""
        return {
            'average_duration': rally_data['duration'].mean(),
            'duration_distribution': rally_data['duration'].describe().to_dict(),
            'common_endings': rally_data['ending_type'].value_counts().to_dict()
        }
    
    def _analyze_court_usage(self, shot_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze court usage patterns."""
        # Create court grid
        grid_size = 2  # feet
        x_bins = np.arange(0, self.court_dimensions['length'] + grid_size, grid_size)
        y_bins = np.arange(0, self.court_dimensions['width'] + grid_size, grid_size)
        
        # Calculate usage heatmap
        H, xedges, yedges = np.histogram2d(
            shot_data['placement_x'],
            shot_data['placement_y'],
            bins=[x_bins, y_bins]
        )
        
        return {
            'usage_heatmap': H.T.tolist(),
            'x_bins': xedges.tolist(),
            'y_bins': yedges.tolist(),
            'hot_zones': self._identify_hot_zones(H)
        }
    
    def _identify_hot_zones(self, heatmap: np.ndarray) -> List[Dict[str, Any]]:
        """Identify hot zones in court usage."""
        # Find local maxima
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(heatmap, size=3)
        hot_zones = []
        
        for i in range(len(heatmap)):
            for j in range(len(heatmap[0])):
                if heatmap[i,j] == local_max[i,j] and heatmap[i,j] > 0:
                    hot_zones.append({
                        'x': j,
                        'y': i,
                        'intensity': float(heatmap[i,j])
                    })
        
        return sorted(hot_zones, key=lambda x: x['intensity'], reverse=True) 